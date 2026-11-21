"""v4.3.1 recheck R16: is the A1 death/alive verdict reliable, and does it bite?

R15 confirmed the 18 candidates are 18 distinct masks, so "18 arms" is honest. Two more
things decide whether SEG-001 rev B is standing on solid ground:

  A. Reproducibility: the A1 report carries `repeat_agreement`. If the same arm re-run
     gives a different verdict, the safe set is noise and the whole task is ill-defined.
  B. Bite: at a selected state, does the recorded (student) action actually die? If the
     student's own action survived, the state would not be one where the student died
     shortly after, and the "requires a decision" claim would weaken.

Also reports, per selected state, how many arms are lethal, so the difficulty of the
task is visible rather than averaged away.

Read-only. Writes only Document/Update/Alpha/v4.3.1/evidence/R16/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
A1 = REPO / "Document" / "Update" / "Alpha" / "v4.2.0" / "evidence" / "T08" / "a1_safe_set_full.json"
MAN = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R12" / "seg001_state_manifest.json"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R16"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a1 = json.loads(A1.read_text(encoding="utf-8"))
    man = json.loads(MAN.read_text(encoding="utf-8"))

    # ---- A. repeat agreement ----
    ra = a1.get("repeat_agreement") or {}
    repeat = {
        "raw": ra,
        "pairs": ra.get("pairs"),
        "agreed": ra.get("agreed"),
        "rate": (ra.get("agreed") / ra.get("pairs")) if ra.get("pairs") else None,
    }

    # ---- B. does the recorded action die? ----
    by_index = {int(s["index"]): s for s in a1["states"]}
    selected = man["train_states"] + man["dev_states"]
    rows = []
    for sel in selected:
        src = by_index[int(sel["index"])]
        recorded = int(src["recorded_action"])
        outcomes = src["outcomes"]
        deaths = sum(1 for v in outcomes.values() if v == "engine_death")
        alive = sum(1 for v in outcomes.values() if v == "engine_alive")
        unknown = sum(1 for v in outcomes.values() if v not in ("engine_death", "engine_alive"))
        rows.append({
            "index": int(sel["index"]),
            "seed": int(sel["seed"]),
            "decision_frame": int(sel["decision_frame"]),
            "offset_to_death": int(sel["offset_to_death"]),
            "recorded_action": recorded,
            "recorded_outcome": outcomes.get(str(recorded)),
            "recorded_dies": outcomes.get(str(recorded)) == "engine_death",
            "student_cand_at_state": int(src.get("student_cand_at_state", -1)),
            "student_matches_recorded": bool(src.get("student_argmax_matches_recorded")),
            "n_death_arms": deaths,
            "n_alive_arms": alive,
            "n_unknown_arms": unknown,
            "safe_set": sel["safe_set"],
            "safe_size": sel["safe_size"],
        })

    n = len(rows)
    rec_dies = sum(1 for r in rows if r["recorded_dies"])
    all_arms_known = all(r["n_unknown_arms"] == 0 for r in rows)
    arms_total = sum(r["n_death_arms"] + r["n_alive_arms"] for r in rows)

    summary = {
        "selected_states": n,
        "recorded_action_dies": rec_dies,
        "recorded_action_dies_rate": (rec_dies / n) if n else None,
        "selected_states_with_any_unknown_arm": sum(1 for r in rows if r["n_unknown_arms"]),
        "all_arms_judged": all_arms_known,
        "arms_judged_over_selected": arms_total,
        "expected_arms_over_selected": n * 18,
        "student_argmax_matches_recorded_true": sum(1 for r in rows if r["student_matches_recorded"]),
        "min_safe_size": min((r["safe_size"] for r in rows), default=None),
        "max_safe_size": max((r["safe_size"] for r in rows), default=None),
        "mean_death_arms": (sum(r["n_death_arms"] for r in rows) / n) if n else None,
    }

    out = {
        "schema": "v431-recheck-r16/1",
        "source": str(A1.relative_to(REPO)),
        "repeat_agreement": repeat,
        "summary": summary,
        "rows": rows,
    }
    (OUT / "verdict_reliability.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("== A. repeat agreement of the A1 arm verdicts ==")
    print(f"  pairs={repeat['pairs']}  agreed={repeat['agreed']}  rate={repeat['rate']}")
    print()
    print("== B. does the recorded (student) action die at the selected states? ==")
    print(f"  selected states           : {summary['selected_states']}")
    print(f"  recorded action dies      : {rec_dies} "
          f"({summary['recorded_action_dies_rate']:.1%})")
    print(f"  states with unknown arms  : {summary['selected_states_with_any_unknown_arm']}")
    print(f"  arms judged / expected    : {summary['arms_judged_over_selected']}"
          f" / {summary['expected_arms_over_selected']}")
    print(f"  safe size range           : {summary['min_safe_size']}..{summary['max_safe_size']}")
    print(f"  mean lethal arms per state: {summary['mean_death_arms']:.2f} / 18")
    print()
    print(f"{'idx':>4s} {'seed':>5s} {'frame':>6s} {'off':>4s} {'rec':>4s} "
          f"{'rec_outcome':>14s} {'|S|':>4s} {'deaths':>7s}")
    print("-" * 66)
    for r in rows[:12]:
        print(f"{r['index']:4d} {r['seed']:5d} {r['decision_frame']:6d} "
              f"{r['offset_to_death']:4d} {r['recorded_action']:4d} "
              f"{str(r['recorded_outcome']):>14s} {r['safe_size']:4d} {r['n_death_arms']:7d}")
    print("   ...")
    print()
    print(f"wrote: {OUT / 'verdict_reliability.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
