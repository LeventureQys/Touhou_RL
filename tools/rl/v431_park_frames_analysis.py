"""v4.3.1 recheck R33 (analysis): PARK is structurally shut, not merely early.

R32 got `ERR park-unsafe ... threads=5 setup=0 null_device=0` at frame 400. Two of those three
could plausibly have been "the stage has not finished setting up yet", which would have reduced
the repair to a single thread-count problem. R33 retried the identical PARK on the real
hash-verified seed-12 tape at frames 400, 1200, 2400, 4800 and 8664.

Result: every attempt is refused with the SAME numbers.

    frame  400  threads=5 setup=0 null_device=0   (100 hash checks ok)
    frame 1200  threads=5 setup=0 null_device=0   (300)
    frame 2400  threads=5 setup=0 null_device=0   (600)
    frame 4800  threads=5 setup=0 null_device=0   (1200)
    frame 8664  threads=5 setup=0 null_device=0   (2166)

So `setup` never reaches GAMEPLAY_SETUP_COMPLETE and `d3dDevice` is never populated, even 8,664
frames into the stage. Those are properties of the shadow build, not of when you ask to park.
The repair is therefore a real code change inside `third_party/th08-rl` (three judgement calls
in `Ready()` / `HandlePark`), which AGENTS.md section 3 requires the user to authorise.

The walk itself is a useful side result: 2,166 recorded hashes checked with 0 mismatches, so the
shadow path reproduces the tape bit-for-bit up to frame 8,664.

Writes only Document/Update/Alpha/v4.3.1/evidence/R33/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R33"
TARGET = EV / "park_frames.json"


def main() -> int:
    if not TARGET.exists():
        print(f"missing {TARGET}", file=sys.stderr)
        return 2
    rec = json.loads(TARGET.read_text(encoding="utf-8"))
    attempts = rec["attempts"]
    triples = {(a["threads"], a["setup"], a["null_device"]) for a in attempts}
    last = attempts[-1]

    rec["analysis"] = {
        "attempts": len(attempts),
        "frames_tried": [a["frame"] for a in attempts],
        "distinct_unsafe_triples": [list(t) for t in sorted(triples)],
        "invariant_across_frames": len(triples) == 1,
        "last_attempt_hash_checks": last["hash_checks_so_far"],
        "last_attempt_hash_mismatches": last["hash_mismatches_so_far"],
        "verdict": {
            "was_it_merely_early": False,
            "structural": True,
            "reading": ("every frame from 400 to 8664 returns threads=5 setup=0 null_device=0, "
                        "so neither `setup` nor `d3dDevice` is a timing effect: the shadow "
                        "build simply never satisfies Ready()/HandlePark"),
            "repair_location": "third_party/th08-rl (Ready() in fork.cpp, HandlePark in fork.cpp)",
            "repair_size": "three judgement calls, not a rewrite",
            "needs_user_approval": True,
        },
        "side_result": {
            "shadow_walk_hash_checks": last["hash_checks_so_far"],
            "shadow_walk_mismatches": last["hash_mismatches_so_far"],
            "meaning": ("the shadow role reproduces the recorded tape exactly to frame 8,664, so "
                        "shadow is a faithful place to run the arms if PARK is ever fixed"),
        },
        "consequence_for_the_plan": {
            "route_A_still_available": True,
            "route_A_cost": ("three gate conditions in one file; recompile rl_v3 changes its "
                             "SHA, so ASSET_SYNC has to be updated (AGENTS.md 8.2)"),
            "route_B_unchanged": "arm A to 200 episodes without touching the engine (3.9 h)",
            "not_attempted_by_design": ["patching third_party", "rebuilding the engine"],
        },
    }
    TARGET.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"frames tried            : {rec['analysis']['frames_tried']}")
    print(f"distinct unsafe triples : {rec['analysis']['distinct_unsafe_triples']}")
    print(f"invariant across frames : {rec['analysis']['invariant_across_frames']}")
    print(f"was it merely early     : {rec['analysis']['verdict']['was_it_merely_early']}")
    print(f"structural              : {rec['analysis']['verdict']['structural']}")
    print(f"shadow walk hash checks : {rec['analysis']['side_result']['shadow_walk_hash_checks']} "
          f"(mismatches {rec['analysis']['side_result']['shadow_walk_mismatches']})")
    print(f"repair location         : {rec['analysis']['verdict']['repair_location']}")
    print(f"needs approval          : {rec['analysis']['verdict']['needs_user_approval']}")
    print()
    print(f"wrote: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
