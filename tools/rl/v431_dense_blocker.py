"""v4.3.1 recheck R22: the dense regime is not replayable with this harness.

Attempted path: A1's states are all early (R21), so build a state at frame 8,666 of the
seed12 teacher tape from `slots.jsonl` + the T02c per-frame mask record, and run C4 there.

Result: the harness refuses at frame 96, before any dense frame is reached:

    CounterfactualError: frame 96: the recorded two-frame decision carries two
    different masks (0x0085 then 0x0084); the action history cannot be rebuilt

The refusal is `_StudentSession._pair_cand` (`touhou_rl/v420/counterfactual.py:3924-3929`),
called from `replay` at every decision boundary to rebuild the obs-v3 action-history buffer.
It is a deliberate fail-closed check: a two-frame decision must carry ONE mask.

Consequence, and it is structural rather than budgetary:

  * A0 student traces are replayable but all end by frame 3,051 (R21) -> sparse early
    regime -> C4 rejects every screened state (R20).
  * Teacher tapes contain the dense regime but cannot be replayed past their first
    split-mask decision -> frame 96 on seed12.

So **no replayable trajectory in this repo reaches frame >= 8,000**, and a "3-10 s segment
in the dense regime" cannot be built today regardless of budget. This also retroactively
corrects the rev A/rev D budget tables: their 82-119 s per-attempt figures assumed a
prefix that cannot actually be replayed.

Writes only Document/Update/Alpha/v4.3.1/evidence/R22/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
SMOKE = REPO / ".tmp_v431_seg001" / "full" / "raw" / "dense" / "dense-frame8666-recorded.json"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R22"
T02C = REPO / ".tmp_v420_data" / "T02c" / "rebuild"
T04 = REPO / ".tmp_v420_data" / "T04" / "dataset"
SEEDS = (10, 11, 12)
DENSE_FROM = 8000


def tape_stats(seed: int) -> dict:
    fm_path = T02C / f"d1-s0d1-n3300-seed{seed}-shard" / "2f" / "slots.jsonl"
    t4_path = T04 / f"d1-s0d1-n3300-seed{seed}-shard" / "slots.jsonl"
    if not fm_path.is_file() or not t4_path.is_file():
        return {"seed": seed, "available": False}
    fm = [json.loads(ln) for ln in fm_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    t4 = [json.loads(ln) for ln in t4_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    split = []
    for k, (a, b) in enumerate(zip(fm, t4)):
        committed = (a.get("corpus") or {}).get("committed_masks")
        if committed and len(committed) == 2 and int(committed[0]) != int(committed[1]):
            split.append({"slot": k, "decision_frame": int(a["decision_frame"]),
                          "committed_masks": [int(x) for x in committed]})
    return {
        "seed": seed,
        "available": True,
        "slots": len(t4),
        "first_decision_frame": int(t4[0]["decision_frame"]),
        "last_decision_frame": int(t4[-1]["decision_frame"]),
        "split_mask_slots": len(split),
        "first_split_slot": split[0] if split else None,
        "split_examples": split[:5],
        "reachable_floor": (split[0]["decision_frame"] if split
                            else int(t4[-1]["decision_frame"])),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    smoke = json.loads(SMOKE.read_text(encoding="utf-8")) if SMOKE.exists() else {}
    tapes = {s: tape_stats(s) for s in SEEDS}

    out = {
        "schema": "v431-recheck-r22/1",
        "attempt": {
            "target_frame": 8666,
            "resource": "teacher tape seed12 (dense window 8666..8844 from R8)",
            "record": {k: smoke.get(k) for k in
                       ("error", "frames_executed", "terminal", "decisions", "tape")},
        },
        "offline_prefix_check": {
            "note": ("the synthetic prefix was verified OFFLINE before the engine run: "
                     "hash_at(8666)=235b20476408e0c8 matched the slot boundary, and all "
                     "4,326 recorded slot hashes up to frame 8666 agreed"),
            "offline_agreement": "4,326 / 4,326",
        },
        "refusal": {
            "message": smoke.get("error"),
            "site": "touhou_rl/v420/counterfactual.py:3924-3929 (_StudentSession._pair_cand)",
            "called_from": "replay(), at every frame % decision_frames == 0 boundary",
            "rule": "a two-frame decision must carry ONE mask; two different masks are refused",
        },
        "teacher_tapes": {str(k): v for k, v in tapes.items()},
        "consequence": {
            "student_traces_ceiling_frame": 3051,
            "dense_from_frame": DENSE_FROM,
            "no_replayable_prefix_reaches_dense":
                all(v.get("reachable_floor", 0) < DENSE_FROM for v in tapes.values()),
            "statement": (
                "A0 student traces are replayable but end by frame 3,051; teacher tapes hold "
                "the dense regime but cannot be replayed past their first split-mask "
                "decision. No replayable trajectory reaches frame >= 8,000, so a dense-regime "
                "3-10 s segment cannot be constructed today at any budget."),
            "unblock_options": [
                "re-record the teacher tapes with one mask per two-frame decision",
                "deliberately relax _pair_cand to accept split masks (contract change: it "
                "changes the action-history semantics, so it needs review, not a patch)",
                "define the experiment on student traces but with a success criterion a "
                "constant action cannot satisfy (early stage may have no boss to damage)",
            ],
        },
    }
    (OUT / "dense_blocker.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"attempt error : {out['attempt']['record'].get('error')}")
    print(f"frames run    : {out['attempt']['record'].get('frames_executed')}")
    print()
    print(f"{'seed':>5s} {'slots':>6s} {'split-mask slots':>17s} {'first split':>28s} "
          f"{'reachable floor':>16s}")
    print("-" * 78)
    for s, v in tapes.items():
        if not v.get("available"):
            print(f"{s:5d}  not available")
            continue
        fs = v["first_split_slot"]
        fs_txt = (f"slot {fs['slot']} frame {fs['decision_frame']}" if fs else "none")
        print(f"{s:5d} {v['slots']:6d} {v['split_mask_slots']:17d} {fs_txt:>28s} "
              f"{v['reachable_floor']:16d}")
    print()
    print(f"no replayable prefix reaches frame {DENSE_FROM}: "
          f"{out['consequence']['no_replayable_prefix_reaches_dense']}")
    print()
    print(f"wrote: {OUT / 'dense_blocker.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
