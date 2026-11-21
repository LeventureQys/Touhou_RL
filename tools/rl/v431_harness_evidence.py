"""v4.3.1 recheck R19: harness validation, and a correction to the safe-set semantics.

Three scripted-policy episodes drive the real engine from frozen state 0 (seed 3042,
decision_frame 2010, expected_hash 6cade82be08f8a95). They answer three different
questions and the answers do not all go the way rev C assumed.

  1. `recorded` — replay the tape's own mask every decision. A1 recorded the student's
     own action dying at frame 2041. This driver steps frames itself (it never calls
     `_run_student_arm`), so reproducing 2041 is a genuine cross-check of the harness.
  2. `const:2`  — hold candidate 2. A1's table says candidate 2 is `engine_alive` at
     horizon 120. A1 measured "cand 2 for 2 frames, then the frozen student for 118".
     If holding cand 2 dies, then A1's safe set S is *continuation-conditional* and is
     NOT a set of survivable action sequences.
  3. `const:7`  — hold candidate 7, another member of the same S.

The last one decides whether this state is admissible as a "requires a combat decision"
segment under the review's own rule (a constant key press must not finish it).
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / ".tmp_v431_seg001" / "full" / "raw"
A1 = REPO / "Document" / "Update" / "Alpha" / "v4.2.0" / "evidence" / "T08" / "a1_safe_set_full.json"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R19"
TAGS = ("recorded", "const2", "const7")


def load(tag: str) -> dict:
    return json.loads((RAW / f"episode-{tag}.json").read_text(encoding="utf-8"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a1 = json.loads(A1.read_text(encoding="utf-8"))
    s0 = a1["states"][0]
    a1_outcomes = {int(k): v for k, v in s0["outcomes"].items()}
    a1_arms = {int(x["cand"]): x for x in (s0.get("arms") or [])}

    episodes = {t: load(t) for t in TAGS}

    recorded = episodes["recorded"]
    const2 = episodes["const2"]
    const7 = episodes["const7"]

    a1_recorded_action = int(s0["recorded_action"])
    a1_recorded_death = (a1_arms.get(a1_recorded_action) or {}).get("first_death_frame")

    checks = {
        "restore_gate_all_ok": all(
            e["restore"].get("matches_frozen") and e["restore"].get("hash_mismatches") == 0
            for e in episodes.values()),
        "recorded_reproduced_a1_death_frame":
            recorded["first_death_frame"] == a1_recorded_death,
        "a1_recorded_action": a1_recorded_action,
        "a1_recorded_death_frame": a1_recorded_death,
        "probe_recorded_death_frame": recorded["first_death_frame"],
    }

    # --- the semantics finding ---
    a1_says_cand2 = a1_outcomes.get(2)
    semantics = {
        "a1_safe_set": [int(x) for x in s0["safe_set"]],
        "a1_definition": ("an arm is `engine_alive` when the candidate's 2 frames are "
                          "followed by the FROZEN STUDENT's continuation to the horizon"),
        "a1_says_cand2": a1_says_cand2,
        "holding_cand2_terminal": const2["terminal"],
        "holding_cand2_death_offset": const2["first_death_offset"],
        "holding_cand7_terminal": const7["terminal"],
        "holding_cand7_frames": const7["frames_executed"],
        "verdict": None,
    }
    continuation_conditional = (
        a1_says_cand2 == "engine_alive"
        and const2["terminal"] == "first_death"
        and (const2["first_death_offset"] or 10 ** 9) <= 120)
    semantics["verdict"] = ("S is continuation-conditional: membership in S does not "
                            "mean the action can be held"
                            if continuation_conditional else
                            "no contradiction observed")

    # --- is this state admissible as a 3 s segment? ---
    constant_action_survives = const7["terminal"] is None and bool(const7["alive_180"])
    admissibility = {
        "rule": ("the review requires a segment that a constant key press cannot finish; "
                 "a constant action surviving the full 180 frames disqualifies it"),
        "constant_cand": 7,
        "constant_survived_180_frames": bool(constant_action_survives),
        "state_0_admissible_at_180_frames": not constant_action_survives,
        "state_0_admissible_at_120_frames": None,
        "note": ("at 120 frames, A1 already answers this: its own recorded action dies at "
                 "offset 31, so a constant action *does* fail there - but 'there' is the "
                 "student-continuation protocol, not this driver's 180-frame protocol"),
    }

    cost = {}
    for tag, e in episodes.items():
        st = e.get("session_stats") or {}
        cost[tag] = {
            "wall_s": e["wall_s"],
            "frames_executed": e["frames_executed"],
            "prefix_frames": st.get("prefix_frames"),
            "prefix_s": st.get("prefix_s"),
            "branch_frames": st.get("branch_frames"),
            "encode_s": st.get("encode_s"),
            "reset_s": st.get("reset_s"),
        }

    out = {
        "schema": "v431-recheck-r19/1",
        "state": {"index": 0, "seed": int(s0["seed"]),
                  "decision_frame": int(s0["decision_frame"]),
                  "expected_hash": s0["expected_hash"],
                  "a1_safe_set": [int(x) for x in s0["safe_set"]]},
        "harness_checks": checks,
        "episodes": {
            "recorded": {k: recorded[k] for k in
                         ("policy_spec", "decisions", "frames_executed", "terminal",
                          "first_death_frame", "first_death_offset", "alive_120",
                          "alive_180", "wall_s", "error")},
            "const2": {k: const2[k] for k in
                       ("policy_spec", "decisions", "frames_executed", "terminal",
                        "first_death_frame", "first_death_offset", "alive_120",
                        "alive_180", "terminal", "wall_s", "error")},
            "const7": {k: const7[k] for k in
                       ("policy_spec", "decisions", "frames_executed", "terminal",
                        "first_death_frame", "first_death_offset", "alive_120",
                        "alive_180", "wall_s", "error")},
        },
        "safe_set_semantics": semantics,
        "segment_admissibility": admissibility,
        "cost": cost,
        "restore": {t: e["restore"] for t, e in episodes.items()},
    }

    (OUT / "harness_validation.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("== harness checks ==")
    for k in ("restore_gate_all_ok", "recorded_reproduced_a1_death_frame",
              "a1_recorded_death_frame", "probe_recorded_death_frame"):
        print(f"  {k:40s} {checks[k]}")
    print()
    print("== episodes (state 0, seed 3042, frame 2010) ==")
    print(f"{'policy':10s} {'frames':>7s} {'terminal':>13s} {'death@':>14s} "
          f"{'offset':>7s} {'alive120':>9s} {'alive180':>9s} {'wall_s':>7s}")
    print("-" * 86)
    for tag in TAGS:
        e = episodes[tag]
        print(f"{e['policy_spec']:10s} {e['frames_executed']:7d} "
              f"{str(e['terminal']):>13s} {str(e['first_death_frame']):>14s} "
              f"{str(e['first_death_offset']):>7s} {str(e['alive_120']):>9s} "
              f"{str(e['alive_180']):>9s} {e['wall_s']:7.2f}")
    print()
    print("== safe-set semantics ==")
    print(f"  A1 safe set              : {semantics['a1_safe_set']}")
    print(f"  A1 says cand 2           : {semantics['a1_says_cand2']}")
    print(f"  holding cand 2           : {semantics['holding_cand2_terminal']} "
          f"at offset {semantics['holding_cand2_death_offset']}")
    print(f"  holding cand 7           : {semantics['holding_cand7_terminal']} "
          f"after {semantics['holding_cand7_frames']} frames")
    print(f"  VERDICT                  : {semantics['verdict']}")
    print()
    print("== admissibility as a 180-frame segment ==")
    print(f"  constant cand 7 survived 180 frames : "
          f"{admissibility['constant_survived_180_frames']}")
    print(f"  state 0 admissible                 : "
          f"{admissibility['state_0_admissible_at_180_frames']}")
    print()
    print(f"wrote: {OUT / 'harness_validation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
