"""v4.3.1 recheck R10: read the existing engine-verified safe-set pass (v4.2.0 T08 A1).

Why this matters for the v4.3.0 review
--------------------------------------
The review asks for a segment that "does not let a constant key press finish it". There
already exists, in this repo, an engine-measured answer to exactly that question: the
v4.2.0 T08 A1 pass ran all 18 first actions from each of 200 frozen states and recorded
which of them survived a 120-frame continuation. A state with `0 < |S| < 18` is a state
where some actions die and some live, i.e. a state that demonstrably requires a decision.

The same file also contains the real engine cost of that pass (prefix frames vs branch
frames vs wall seconds), which is a measurement rather than the estimate the v4.3.1
budget currently rests on.

This script only reads and summarises. It does not launch the engine.

Writes only new files under Document/Update/Alpha/v4.3.1/evidence/R10/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "Document" / "Update" / "Alpha" / "v4.2.0" / "evidence" / "T08" / "a1_safe_set_full.json"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R10"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(SRC.read_text(encoding="utf-8"))

    out = {
        "schema": "v431-recheck-r10/1",
        "source": str(SRC.relative_to(REPO)),
        "source_schema": d.get("schema"),
        "status": d.get("status"),
        "exit_code": d.get("exit_code"),
        "horizon_frames": d.get("horizon_frames"),
        "candidates": d.get("candidates"),
        "states_total": d.get("states_total"),
        "arms_total_expected": d.get("arms_total_expected"),
        "arms_present": d.get("arms_present"),
        "counts": d.get("counts"),
        "safe_set": d.get("safe_set"),
        "discriminating": d.get("discriminating"),
        "proper": d.get("proper"),
        "per_candidate": d.get("per_candidate"),
        "per_offset": d.get("per_offset"),
        "selection": d.get("selection"),
        "identity": d.get("identity"),
        "cost": {
            "branch_frames": d.get("branch_frames"),
            "prefix_frames": d.get("prefix_frames"),
            "student_forwards": d.get("student_forwards"),
            "outer_branch_frame_budget": d.get("outer_branch_frame_budget"),
            "wall": d.get("wall"),
        },
        "cannot_claim": d.get("cannot_claim"),
        "notes": d.get("notes"),
    }

    # ---- derived cost facts ----
    w = d.get("wall") or {}
    bf = d.get("branch_frames") or 0
    pf = d.get("prefix_frames") or 0
    arms = d.get("arms_present") or 1
    states = d.get("states_total") or 1
    tot = w.get("in_process_total_s")
    out["derived"] = {
        "prefix_share_of_frames": round(pf / (pf + bf), 4) if (pf + bf) else None,
        "branch_share_of_frames": round(bf / (pf + bf), 4) if (pf + bf) else None,
        "prefix_frames_per_arm": round(pf / arms, 1) if arms else None,
        "branch_frames_per_arm": round(bf / arms, 1) if arms else None,
        "arms_per_state": arms / states if states else None,
        "wall_s_per_arm": round(tot / arms, 3) if (tot and arms) else None,
        "engine_frames_per_s": round((pf + bf) / tot, 1) if (tot and (pf + bf)) else None,
        "wall_h_for_200_episodes_3s": None,
    }
    # cost of one 3 s segment attempt at this engine's measured rate, using the
    # measured prefix cost per arm from this pass as the reference
    fps = out["derived"]["engine_frames_per_s"]
    if fps:
        for seg_frames, label in ((180, "3s"), (300, "5s"), (600, "10s")):
            # an attempt must replay a prefix; use the measured per-arm prefix
            per_attempt_frames = (out["derived"]["prefix_frames_per_arm"] or 0) + seg_frames
            out["derived"][f"attempt_frames_{label}"] = round(per_attempt_frames, 1)
            out["derived"][f"attempt_wall_s_{label}"] = round(per_attempt_frames / fps, 1)
            out["derived"][f"wall_h_200_attempts_{label}"] = round(
                200 * per_attempt_frames / fps / 3600, 2)

    (OUT / "safe_set_evidence.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- console ----
    ss = out["safe_set"] or {}
    print(f"source: {out['source']}  schema={out['source_schema']}  status={out['status']}")
    print(f"states={out['states_total']} arms_expected={out['arms_total_expected']} "
          f"arms_present={out['arms_present']} horizon={out['horizon_frames']} frames")
    print()
    print("== safe-set size histogram ==")
    print(json.dumps(ss.get("histogram_0_18"), ensure_ascii=False))
    print(f"  sizes buckets: {json.dumps(ss.get('buckets'), ensure_ascii=False)[:400]}")
    print(f"  ratios_all_states: {json.dumps(ss.get('ratios_all_states'), ensure_ascii=False)[:300]}")
    print()
    disc = out["discriminating"] or {}
    print("== discriminating (0 < |S| < 18) ==")
    print(f"  definition: {disc.get('definition')}")
    print(f"  count={disc.get('complete_states_with_0_lt_S_lt_18')} "
          f"of {disc.get('complete_states')} complete; "
          f"rate_over_all={disc.get('rate_over_all_states')} "
          f"rate_over_complete={disc.get('rate_over_complete_states')}")
    pr = out["proper"] or {}
    print(f"  proper count={pr.get('count')}/{pr.get('denominator')} rate={pr.get('rate')} "
          f"wilson95={pr.get('wilson_95')} floor={pr.get('gate_floor')} "
          f"met={pr.get('gate_floor_met')}")
    print()
    print("== measured engine cost of this pass ==")
    for k, v in out["derived"].items():
        print(f"  {k} = {v}")
    print()
    print("== outcome counts ==")
    print(f"  {json.dumps(out['counts'], ensure_ascii=False)[:400]}")
    print()
    print("== cannot_claim ==")
    for c in (out["cannot_claim"] or []):
        print(f"  - {c}")
    print()
    print(f"wrote: {OUT / 'safe_set_evidence.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
