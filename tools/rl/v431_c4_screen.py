"""v4.3.1 recheck R20: the C4 calibration gate result.

Reads the per-(state, candidate) episodes produced by `.tmp_v431_seg001/run_c4.sh` and
decides, per state, whether it is admissible as a "requires a combat decision" segment:

    admissible  <=>  no constant action survives the whole segment

Also derives the `C-const-best` baseline (the constant action that survives longest),
which rev D uses as the second gate baseline alongside `C-fair`.

The output states the pass rate explicitly, because the honest headline may well be
"most A1 states are NOT admissible at 3 s" - which is a finding, not a failure.

Writes only Document/Update/Alpha/v4.3.1/evidence/R20/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
C4 = REPO / ".tmp_v431_seg001" / "full" / "raw" / "c4"
MAN = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R12" / "seg001_state_manifest.json"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R20"
N_CAND = 18
SEGMENT = 180


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads(MAN.read_text(encoding="utf-8"))
    split_of = {}
    for s in man["train_states"]:
        split_of[int(s["index"])] = "train"
    for s in man["dev_states"]:
        split_of[int(s["index"])] = "dev"

    files = sorted(C4.glob("c4-state*-cand*.json"))
    if not files:
        print(f"no C4 episodes under {C4}", file=sys.stderr)
        return 2

    per_state: dict[int, dict] = {}
    errors = []
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        st = int(rec["state_index"])
        cand = int(rec["policy_spec"].split(":", 1)[1])
        if rec.get("error"):
            errors.append({"file": f.name, "error": rec["error"]})
            continue
        entry = per_state.setdefault(st, {"seed": rec["seed"],
                                          "decision_frame": rec["decision_frame"],
                                          "const": {}})
        entry["const"][cand] = {
            "terminal": rec["terminal"],
            "first_death_frame": rec["first_death_frame"],
            "first_death_offset": rec["first_death_offset"],
            "alive_120": rec["alive_120"],
            "alive_180": rec["alive_180"],
            "frames_executed": rec["frames_executed"],
            "wall_s": rec["wall_s"],
        }

    states_out = []
    for st in sorted(per_state):
        e = per_state[st]
        const = e["const"]
        complete = len(const) == N_CAND
        survivors = sorted(c for c, v in const.items() if v["terminal"] is None
                           and v["alive_180"])
        # survival length: a non-terminal run is the full segment
        def survival(c):
            v = const[c]
            return SEGMENT if v["terminal"] is None else int(v["frames_executed"])
        best = max(const, key=survival) if const else None
        states_out.append({
            "state_index": st,
            "split": split_of.get(st),
            "seed": e["seed"],
            "decision_frame": e["decision_frame"],
            "complete_18": complete,
            "n_constants_run": len(const),
            "n_survivors_180": len(survivors),
            "surviving_candidates": survivors,
            "admissible": bool(complete and len(survivors) == 0),
            "best_constant": best,
            "best_constant_survival_frames": survival(best) if best is not None else None,
            "best_constant_survival_s": round(survival(best) * 2 / 60.0, 2) if best is not None else None,
            "const_detail": {str(c): const[c] for c in sorted(const)},
        })

    n = len(states_out)
    adm = [s for s in states_out if s["admissible"]]
    adm_train = [s for s in adm if s["split"] == "train"]
    adm_dev = [s for s in adm if s["split"] == "dev"]

    out = {
        "schema": "v431-recheck-r20/1",
        "segment_frames": SEGMENT,
        "segment_s": round(SEGMENT * 2 / 60.0, 2),
        "rule": ("a state is admissible iff none of the 18 constant actions survives the "
                 "whole segment; C-const-best is the longest-surviving constant"),
        "counts": {
            "states_screened": n,
            "states_complete": sum(1 for s in states_out if s["complete_18"]),
            "admissible": len(adm),
            "admissible_train": len(adm_train),
            "admissible_dev": len(adm_dev),
            "admissible_rate": (len(adm) / n) if n else None,
            "errors": len(errors),
        },
        "gate": {
            "S1b_min_required_each_side": 4,
            "S1b_train_ok": len(adm_train) >= 4,
            "S1b_dev_ok": len(adm_dev) >= 4,
            "S1b_triggered": not (len(adm_train) >= 4 and len(adm_dev) >= 4),
        },
        "admissible_states": [
            {k: s[k] for k in ("state_index", "split", "seed", "decision_frame",
                               "n_survivors_180")} for s in adm],
        "states": states_out,
        "errors": errors,
    }
    (OUT / "c4_screen.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    c = out["counts"]
    print(f"states screened          : {c['states_screened']} "
          f"(complete 18/18: {c['states_complete']})")
    print(f"admissible               : {c['admissible']} "
          f"(train {c['admissible_train']}, dev {c['admissible_dev']}) "
          f"rate {c['admissible_rate']:.3f}" if c['admissible_rate'] is not None else "")
    print(f"errors                   : {c['errors']}")
    print(f"S1b (< 4 admissible/side) : {out['gate']['S1b_triggered']}")
    print()
    hdr = (f"{'state':>6s} {'split':>6s} {'18/18':>6s} {'survivors':>10s} "
           f"{'admissible':>11s} {'best const':>11s} {'best surv':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for s in states_out:
        print(f"{s['state_index']:6d} {str(s['split']):>6s} {str(s['complete_18']):>6s} "
              f"{s['n_survivors_180']:10d} {str(s['admissible']):>11s} "
              f"{str(s['best_constant']):>11s} "
              f"{(str(s['best_constant_survival_frames']) + 'f'):>10s}")
    print()
    if adm:
        print("ADMISSIBLE:", [(s["state_index"], s["split"]) for s in adm])
    else:
        print("ADMISSIBLE: none — every screened state can be finished by a constant action")
    print()
    print(f"wrote: {OUT / 'c4_screen.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
