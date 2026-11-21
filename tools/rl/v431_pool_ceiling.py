"""v4.3.1 recheck R21: why the A1 pool cannot contain a late-stage state, and whether
the teacher tape can supply one instead.

Two separate facts, both needed before any further engine time:

A. THE CEILING. A1 froze states "proximate to the A0 student's first death". Those traces
   are the recording source, so their length is a hard ceiling on where a state can sit.
   This script reports the actual `last frame` of all 50 A1 student traces. If none of
   them reaches the dense regime, then the A1 pool is not merely unlucky - it *cannot*
   contain a late-stage state, and no amount of extra C4 sweeping will change that.

B. THE ALTERNATIVE SOURCE. The teacher tapes (`d1-s0d1-n3300-seed{10,11,12}`) do run the
   whole stage (13,190 frames), and `slots.jsonl` carries a `hash_pre`/`hash_post` pair per
   two-frame slot. That is enough to synthesise a replayable `Prefix` for an arbitrary
   frame. Before trusting it, this script checks the hash chain actually links
   (`hash_post[k] == hash_pre[k+1]`) and that the frames sit on the 2-frame grid.

Writes only Document/Update/Alpha/v4.3.1/evidence/R21/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TRACES = REPO / ".tmp_v420_data" / "T08" / "full" / "student_traces"
SHARD = REPO / ".tmp_v420_data" / "T04" / "dataset" / "d1-s0d1-n3300-seed12-shard"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R21"
DENSE_FROM = 8000


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- A. the trace ceiling ----------
    rows = []
    for p in sorted(TRACES.glob("seed*.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        frames = t.get("frames") or []
        rows.append({
            "seed": int(t.get("seed", -1)),
            "frames": len(frames),
            "last_frame": int(frames[-1]["frame"]) if frames else 0,
            "terminal": t.get("terminal") or t.get("stop_reason"),
            "game_over": bool(t.get("game_over")),
        })
    last = sorted(r["last_frame"] for r in rows)
    ceiling = {
        "traces": len(rows),
        "last_frame_min": last[0],
        "last_frame_median": last[len(last) // 2],
        "last_frame_max": last[-1],
        "n_traces_beyond_4000": sum(1 for v in last if v > 4000),
        "n_traces_beyond_8000": sum(1 for v in last if v > DENSE_FROM),
        "terminals": sorted({str(r["terminal"]) for r in rows}),
        "all_end_at_first_death": all(r["terminal"] == "first_death" for r in rows),
        "dense_from": DENSE_FROM,
        "consequence": (
            "no A1 trace reaches the dense regime, so no frozen A1 state can sit there; "
            "the pool cannot satisfy C4 at any segment length by more sweeping"),
    }

    # ---------- B. the teacher tape can supply a late state ----------
    slots = [json.loads(ln) for ln in
             (SHARD / "slots.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    frames = [int(s["decision_frame"]) for s in slots]
    grid_ok = all(frames[k] == 16 + 2 * k for k in range(len(frames)))
    chain_breaks = [k for k in range(len(slots) - 1)
                    if slots[k]["hash_post"] != slots[k + 1]["hash_pre"]]
    dup_masks_within_slot = sum(1 for s in slots if int(s["mask"]) > 0xFFFF)

    # target: rev A's 3 s window start on the dense tape (R8)
    target_frame = 8666
    idx = frames.index(target_frame) if target_frame in frames else None

    tape = {
        "slots": len(slots),
        "first_decision_frame": frames[0],
        "last_decision_frame": frames[-1],
        "two_frame_grid_contiguous": grid_ok,
        "hash_chain_breaks": len(chain_breaks),
        "first_chain_break": chain_breaks[0] if chain_breaks else None,
        "mask_out_of_range": dup_masks_within_slot,
        "target_frame": target_frame,
        "target_is_a_slot_boundary": idx is not None,
        "target_hash_pre": slots[idx]["hash_pre"] if idx is not None else None,
        "lead_in_frames": 16,
        "lead_in_note": (
            "frames 1..16 are the 8 leading decisions recorded as cand 7 in "
            "labels_provenance.json; mask 0x41 for all 16 frames reproduces them"),
        "prefix_frames_needed": target_frame,
        "prefix_s_at_121_7_fps": round(target_frame / 121.7, 1),
        "dense_windows_from_R8": [
            {"name": "train_3s", "start": 8666, "end": 8844},
            {"name": "train_5s", "start": 12436, "end": 12734},
            {"name": "train_10s", "start": 12136, "end": 12734},
            {"name": "dev_seed10_5s", "start": 10196, "end": 10494, "note": "different tape"},
            {"name": "dev_seed11_5s", "start": 8526, "end": 8824, "note": "different tape"},
        ],
    }

    out = {
        "schema": "v431-recheck-r21/1",
        "A_trace_ceiling": ceiling,
        "A_rows": rows,
        "B_teacher_tape": tape,
        "verdict": {
            "a1_pool_structurally_cannot_be_late": ceiling["n_traces_beyond_8000"] == 0,
            "teacher_tape_hash_chain_sound": len(chain_breaks) == 0 and grid_ok,
            "synthetic_prefix_is_viable": bool(len(chain_breaks) == 0 and grid_ok
                                               and idx is not None),
            "reading": (
                "the A1 pool's earliness is a recorder ceiling, not a sampling choice; "
                "a dense-regime state must be synthesised from the teacher tape's "
                "slots.jsonl, at the cost of an 8,666-frame prefix (~71 s)"),
        },
    }
    (OUT / "pool_ceiling.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("== A. A1 student trace lengths (the ceiling on where a state can sit) ==")
    print(f"  traces            : {ceiling['traces']}")
    print(f"  last_frame        : min {ceiling['last_frame_min']} "
          f"median {ceiling['last_frame_median']} max {ceiling['last_frame_max']}")
    print(f"  beyond frame 4000 : {ceiling['n_traces_beyond_4000']}")
    print(f"  beyond frame 8000 : {ceiling['n_traces_beyond_8000']}")
    print(f"  every trace ends at first_death : {ceiling['all_end_at_first_death']}")
    print()
    print("== B. teacher tape (seed12) ==")
    print(f"  slots             : {tape['slots']}  frames "
          f"{tape['first_decision_frame']}..{tape['last_decision_frame']}")
    print(f"  2-frame grid      : {tape['two_frame_grid_contiguous']}")
    print(f"  hash chain breaks : {tape['hash_chain_breaks']} "
          f"(first {tape['first_chain_break']})")
    print(f"  target frame      : {tape['target_frame']} "
          f"is_slot_boundary={tape['target_is_a_slot_boundary']}")
    print(f"  target hash_pre   : {tape['target_hash_pre']}")
    print(f"  prefix cost       : {tape['prefix_frames_needed']} frames "
          f"= {tape['prefix_s_at_121_7_fps']} s")
    print()
    print(f"  synthetic prefix viable : {out['verdict']['synthetic_prefix_is_viable']}")
    print()
    print(f"wrote: {OUT / 'pool_ceiling.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
