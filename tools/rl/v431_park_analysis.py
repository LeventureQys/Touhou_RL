"""v4.3.1 recheck R32 (analysis): why PARK refuses, exactly.

The probe proved three things and one blocker:

  works : shadow role starts headless (DISPLAY unset), RESET returns ready=1 and
          hash=83d953750afaea5a -- the same reset hash the S1c seed-12 tape records, so the
          shadow path reproduces the same initial state;
  works : single-frame stepping to frame 400 works in shadow mode;
  BLOCKED: `PARK` answers
          `ERR park-unsafe role=shadow-required threads=5 setup=0 null_device=0`

That single line names three independent unmet preconditions, and `Ready()` in fork.cpp plus
`HandlePark`'s own guard are what demand them:

  threads=5   Ready() requires ThreadCount() == 1 (it reads /proc/self/task). The engine
              brings up 5 threads even with SDL_AUDIODRIVER=dummy.
  setup=0     Ready() requires gameplaySetupState == GAMEPLAY_SETUP_COMPLETE; at frame 400 the
              shadow's setup state is still 0.
  null_device=0
              HandlePark requires g_Supervisor.d3dDevice != NULL. The null-device shim
              (null_device.cpp) only checks that DISPLAY is empty -- it does not populate a
              D3D device, so `d3dDevice` stays NULL precisely in the mode that is allowed to
              fork. That is a contradiction inside the harness, not a configuration error.

So the fork-server is real, documented, and **unreachable in this build**. Earlier I wrote
that a save-state "needs a fork-server to be written"; the accurate statement is that the
fork-server exists and its gate has to be repaired -- still a third_party change, and per
AGENTS.md section 3 that needs the user's approval.

This script folds the analysis into the R32 evidence file.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R32"
TARGET = EV / "park_spawn.json"


def main() -> int:
    if not TARGET.exists():
        print(f"missing {TARGET}", file=sys.stderr)
        return 2
    rec = json.loads(TARGET.read_text(encoding="utf-8"))
    rec["analysis"] = {
        "what_works": {
            "shadow_starts_headless": True,
            "reset_hash": ("83d953750afaea5a", "matches the S1c seed-12 tape reset hash"),
            "stepping_to_frame": rec.get("steps", {}).get("warmup_frame"),
        },
        "refusal": "ERR park-unsafe role=shadow-required threads=5 setup=0 null_device=0",
        "three_blockers": [
            {"name": "threads=5", "requires": "ThreadCount() == 1 (Ready(), fork.cpp)",
             "observed": 5,
             "note": "the engine brings up 5 threads even with SDL_AUDIODRIVER=dummy"},
            {"name": "setup=0", "requires": "gameplaySetupState == GAMEPLAY_SETUP_COMPLETE",
             "observed": 0, "note": "shadow setup state is still 0 at frame 400"},
            {"name": "null_device=0",
             "requires": "g_Supervisor.d3dDevice != NULL (HandlePark)",
             "observed": 0,
             "note": ("null_device.cpp only requires DISPLAY to be empty; it never populates a "
                      "D3D device, so d3dDevice is NULL exactly in the mode allowed to fork -- "
                      "a contradiction inside the harness")},
        ],
        "verdict": {
            "fork_server_exists": True,
            "fork_server_usable_here": False,
            "why": "all three Ready()/HandlePark preconditions fail in this build",
            "correction_to_earlier_claim": (
                "rev L said a save-state 'needs a fork-server written'; the fork-server already "
                "exists (PARK/SPAWN/CHILD with documented replies) and what is needed is a "
                "repair of its gate"),
            "still_needs": ("a change inside third_party/th08-rl (Ready()/HandlePark gate), "
                            "which per AGENTS.md section 3 requires the user's approval"),
            "not_attempted": ["rebuilding the engine", "patching fork.cpp", "CRIU"],
        },
        "reusable_assets": {
            "probe": "tools/rl/v431_park_spawn.py",
            "runner": ".tmp_v431_seg001/run_park_spawn.sh",
            "protocol": rec.get("protocol"),
        },
    }
    TARGET.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("== what works ==")
    print("  shadow role starts headless, RESET ready=1, hash=83d953750afaea5a "
          "(matches the S1c seed-12 tape)")
    print(f"  stepped to frame {rec['steps'].get('warmup_frame')}")
    print()
    print("== PARK refusal ==")
    print(f"  {rec['analysis']['refusal']}")
    for b in rec["analysis"]["three_blockers"]:
        print(f"  - {b['name']:16s} requires {b['requires']}")
        print(f"    {'':16s} {b['note']}")
    print()
    v = rec["analysis"]["verdict"]
    print(f"fork server exists : {v['fork_server_exists']}")
    print(f"usable here        : {v['fork_server_usable_here']}")
    print(f"correction         : {v['correction_to_earlier_claim']}")
    print(f"still needs        : {v['still_needs']}")
    print()
    print(f"wrote: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
