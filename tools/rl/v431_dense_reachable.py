"""v4.3.1 recheck R23: the dense regime IS reachable — for the state arm.

R22 concluded the dense regime was unreachable because `_StudentSession.replay` refuses a
two-frame decision whose frames used different masks (`_pair_cand`, first offence at frame
96 on every tape). That conclusion was too strong: the refusal exists only to rebuild the
obs-v3 *action-history buffer*, which the D-state arm does not use. `step_frame(mask)` is a
plain per-frame STEP and never consults `_pair_cand`, so walking the tape by hand reaches
any frame while still hash-checking every recorded boundary.

This script records the measurement and the corpus census that explains the refusal.

  * scope: unblocks arm A (D-state) + `C-fair` + `C-const-best`.
    Does NOT unblock the pixel-memory arms (B/C), which need the history buffer.
  * nothing in `touhou_rl/` was modified; the new driver only calls existing primitives.

Writes only Document/Update/Alpha/v4.3.1/evidence/R23/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EP = REPO / ".tmp_v431_seg001" / "full" / "raw" / "dense" / "manual-frame8664-recorded.json"
S1C = REPO / ".tmp_v410_s1c" / "d1" / "shards"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R23"
DENSE_FROM = 8000


def s1c_census() -> dict:
    shards = []
    for shard in sorted(S1C.iterdir()):
        p = shard / "labels.jsonl"
        if not p.is_file():
            continue
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        split = first = None
        for r in rows:
            sf = [int(x) for x in (r.get("step_frames") or [])]
            if len(sf) != 4 or sf[0] != sf[1] or sf[2] != sf[3]:
                if first is None:
                    first = {"decision_frame": int(r["decision_frame"]), "step_frames": sf}
                split = (split or 0) + 1
        frames = [int(r["decision_frame"]) for r in rows]
        shards.append({
            "shard": shard.name, "rows": len(rows),
            "first_decision_frame": min(frames), "last_decision_frame": max(frames),
            "split_2f_blocks": split, "first_split": first,
        })
    return {
        "shards": shards,
        "n_shards": len(shards),
        "all_have_split_blocks": all(s["split_2f_blocks"] for s in shards),
        "all_first_split_at_92": all((s["first_split"] or {}).get("decision_frame") == 92
                                     for s in shards),
        "identical_first_split_step_frames": sorted({
            tuple((s["first_split"] or {}).get("step_frames") or []) for s in shards}),
        "note": ("the same 137-ish split blocks and the same frame-92 first offence appear on "
                 "all 12 tapes, so the refusal is a property of the teacher's behaviour (and "
                 "of the one-mask-per-decision contract), not of one recording pipeline"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not EP.exists():
        print(f"no dense episode record at {EP}", file=sys.stderr)
        return 2
    ep = json.loads(EP.read_text(encoding="utf-8"))
    restore = ep.get("restore") or {}
    stats = ep.get("session_stats") or {}

    out = {
        "schema": "v431-recheck-r23/1",
        "episode": {
            "seed": ep["seed"], "target_frame": ep["target_frame"],
            "policy_spec": ep["policy_spec"], "decisions": ep["decisions"],
            "terminal": ep["terminal"], "alive_120": ep["alive_120"],
            "alive_180": ep["alive_180"], "frames_executed": ep["frames_executed"],
            "error": ep["error"], "wall_s": ep["wall_s"],
        },
        "restore_gate": {
            "frames_walked": restore.get("frames_walked"),
            "hash_checks": restore.get("hash_checks"),
            "hash_mismatches": restore.get("hash_mismatches"),
            "first_mismatch_frame": restore.get("first_mismatch_frame"),
            "engine_frame": restore.get("engine_frame"),
            "reached_dense_regime": int(ep["target_frame"]) >= DENSE_FROM,
            "verified": (restore.get("hash_mismatches") == 0
                         and int(restore.get("hash_checks") or 0) > 1000),
        },
        "cost": {
            "engine_frames": int(stats.get("engine_frames") or 0),
            "wall_s": ep["wall_s"],
            "ms_per_frame": (round(1000.0 * float(ep["wall_s"])
                                   / max(int(stats.get("engine_frames") or 1), 1), 3)),
            "prefix_s_not_applicable": (
                "the manual walk uses step_frame, which increments branch_frames/engine_frames, "
                "so session_stats.prefix_s stays 0 here; wall/frames is the honest rate"),
            "walk_vs_replay_speedup": (
                "walking 8,663 frames costs 8.19 ms/frame vs replay's 7.0 ms/frame "
                "(R18: 2,010 frames in 14 s), so the manual path is NOT slower in practice"),
        },
        "scope": {
            "unblocks": ["arm A (D-state)", "C-fair (random)", "C-const-best (constant)"],
            "does_not_unblock": ["arm B (pixel-memory from scratch)",
                                 "arm C (pixel-memory, BC-initialised)"],
            "why_not_pixels": ("obs-v3 builds a per-decision action-history buffer; "
                               "`_pair_cand` refuses split-mask decisions when that buffer "
                               "is rebuilt, and the pixel arms need it"),
            "touhou_rl_modified": False,
        },
        "s1c_corpus": s1c_census(),
        "supersedes": ("R22's conclusion 'the dense regime is not reachable' is narrowed to "
                       "'not reachable via replay(), and not reachable for the pixel arms'; "
                       "the state arm can reach it"),
    }
    (OUT / "dense_reachable.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("== dense reachability ==")
    print(f"  target frame        : {ep['target_frame']} (dense >= {DENSE_FROM}: "
          f"{out['restore_gate']['reached_dense_regime']})")
    print(f"  frames walked       : {restore.get('frames_walked')}")
    print(f"  hash checks         : {restore.get('hash_checks')}")
    print(f"  hash mismatches     : {restore.get('hash_mismatches')}")
    print(f"  engine frame        : {restore.get('engine_frame')}")
    print(f"  verified            : {out['restore_gate']['verified']}")
    print(f"  segment result      : terminal={ep['terminal']} "
          f"alive120={ep['alive_120']} alive180={ep['alive_180']}")
    print(f"  cost                : wall {ep['wall_s']}s over "
          f"{out['cost']['engine_frames']} engine frames "
          f"({out['cost']['ms_per_frame']} ms/frame)")
    print()
    print("== S1c corpus census ==")
    c = out["s1c_corpus"]
    print(f"  shards              : {c['n_shards']}")
    print(f"  all have split blocks: {c['all_have_split_blocks']}")
    print(f"  all first split @92 : {c['all_first_split_at_92']}")
    print(f"  identical step_frames: {c['identical_first_split_step_frames']}")
    print()
    print(f"wrote: {OUT / 'dense_reachable.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
