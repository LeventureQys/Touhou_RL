"""v4.3.1 recheck R26: arm A can see bullets — the HAZARDS probe answered.

R25 recorded a real gap: `STATE` has 25 fields and no bullet or boss field, so the prereg's
arm A could not be built as written. `HAZARDS` was the only candidate channel and it is
listed among `CHILD_SNAPSHOT_VERBS`, so it was an open question whether an env-role session
answers it.

It does. Measured on the seed12 tape at frame 8,664 (a dense frame):

    HAZARDS  OK  12778 bytes   nb=183 bullets  nl=0 lasers  ne=4 enemies
    PLAYER   OK    331 bytes   hp/lives/bombs/power/focus
    EVENTS   OK    127 bytes   frame-event bitmask
    HUD      OK    172 bytes
    STATE    OK    249 bytes

So arm A becomes **A-HAZARDS**: the state arm sees the actual bullet geometry. This script
records the measurement plus the decoder entry point and the row field lists, so the arm's
observation contract can be written without re-deriving anything.

Writes only Document/Update/Alpha/v4.3.1/evidence/R26/.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROBE = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R26" / "hazard_probe.json"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R26"


def main() -> int:
    if not PROBE.exists():
        print(f"no probe record at {PROBE}", file=sys.stderr)
        return 2
    probe = json.loads(PROBE.read_text(encoding="utf-8"))

    from touhou_rl.v400.gym import protocol as P

    def fields_of(cls) -> list[str]:
        return [f.name for f in dataclasses.fields(cls)]

    verbs = probe["verbs"]
    answered = {v: bool(e.get("ok")) and not e.get("is_err") for v, e in verbs.items()}

    out = {
        "schema": "v431-recheck-r26/1",
        "question": ("can an env-role session (the only role that can produce obs-v3 "
                     "pixels, so the only role the arms can run in) see bullets?"),
        "probe": {
            "seed": probe["seed"], "frame": probe["frame"],
            "walk": probe["walk"],
            "verbs_answered": answered,
            "hazards_reply_bytes": (verbs.get("HAZARDS") or {}).get("reply_len"),
            "hazards_reply_head": (verbs.get("HAZARDS") or {}).get("reply_head"),
        },
        "answer": {
            "hazards_available_on_env": answered.get("HAZARDS", False),
            "arm_a_choice": "A-HAZARDS",
            "supersedes": ("R25's partB 'status: open'; the arm can be bullet-aware, so the "
                           "R25 option list collapses to A-HAZARDS"),
        },
        "observation_contract_for_arm_a": {
            "source": "session.client.hazards() -> protocol.HazardFrame",
            "decoder": "touhou_rl.v400.gym.protocol.parse_hazards_reply / "
                       "EnvHarnessClient.hazards(limit=0)",
            "hazard_frame_fields": fields_of(P.HazardFrame),
            "bullet_row_fields": fields_of(P.BulletRow),
            "enemy_row_fields": fields_of(P.EnemyRow),
            "laser_row_fields": fields_of(P.LaserRow),
            "engine_totals_seen_at_this_frame": {
                "bullets": 183, "lasers": 0, "enemies": 4},
            "note": ("BulletRow.collision_size is the FULL width; the lethal half-width is "
                     "`half_width`. EnemyRow's lethal half extent is hitboxDimensions/3. "
                     "Getting this wrong is the specific error 11-P1 §2.1 names."),
        },
        "consequence": {
            "arm_a_is_buildable_with_bullets": True,
            "remaining_gap": ("arm A's feature vector is not yet defined; this record only "
                              "establishes that the channel exists and how to decode it"),
            "still_blocked": ("pixel arms B/C: `_pair_cand` refuses split-mask decisions when "
                              "rebuilding the obs-v3 action history (R22/R23)"),
        },
        "caveats": [
            "one frame, one tape: the counts (183 bullets) are a measurement at frame 8,664 of "
            "seed12, not a property of the pattern",
            "the reply is 12.8 KB per query; a 600-frame episode that queries every frame "
            "would add 7.7 MB of protocol traffic, so the arm should query per DECISION",
        ],
    }
    (OUT / "hazards_available.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("== verbs on the env-role session ==")
    for v, e in verbs.items():
        print(f"  {v:8s} answered={answered[v]}  ok={e.get('ok')}  "
              f"len={e.get('reply_len', e.get('error'))}  is_err={e.get('is_err')}")
    print()
    print(f"hazards reply head: {(verbs.get('HAZARDS') or {}).get('reply_head', '')[:120]}")
    print()
    o = out["observation_contract_for_arm_a"]
    print(f"decoder: {o['decoder']}")
    print(f"HazardFrame  : {o['hazard_frame_fields']}")
    print(f"BulletRow    : {o['bullet_row_fields']}")
    print(f"EnemyRow     : {o['enemy_row_fields']}")
    print(f"LaserRow     : {o['laser_row_fields']}")
    print()
    print(f"arm A choice : {out['answer']['arm_a_choice']}")
    print(f"wrote: {OUT / 'hazards_available.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
