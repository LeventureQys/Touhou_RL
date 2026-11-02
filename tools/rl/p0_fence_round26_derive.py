"""Derive the report numbers for P0 round26 from the raw evidence.

Read-only: writes ``derived.json`` next to the evidence.  Every number quoted in
``reports/P0-fence-round26.md`` comes from here or straight from summary.json.
"""

from __future__ import annotations

import json
import statistics as stats
from collections import Counter
from pathlib import Path

EV = Path("Document/Update/Alpha/v3.7.3/evidence/p0-fence-round26")


def load_jsonl(name):
    path = EV / name
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    summary = json.loads((EV / "summary.json").read_text(encoding="utf-8"))
    trace = load_jsonl("trace.jsonl")
    samples = load_jsonl("samples.jsonl")
    legacy = load_jsonl("legacy.jsonl")
    trials_raw = json.loads((EV / "trials.json").read_text(encoding="utf-8"))["raw"]
    practice = summary["sessions"]["practice"]
    demo = summary["sessions"]["demo"]

    derived = {
        "build": summary["build"],
        "environment": summary["environment"],
        "demo": {
            "frames": demo["phases"]["M1_clock"]["frames"],
            "presentHz": demo["phases"]["M1_clock"]["presentHz"],
            "holdUsMedian": stats.median([r["hold_last_us"] for r in trace]),
            "holdUsMax": max(r["hold_last_us"] for r in trace),
            "bulletsMin": min(r["active_bullets"] for r in trace),
            "bulletsMax": max(r["active_bullets"] for r in trace),
            "playerStates": demo["phases"]["M1_clock"]["playerStates"],
            "fenceFinal": demo.get("fenceFinal"),
        },
        "samples": {
            "accepted": len(samples),
            "distinctTickets": len({s["ticket"] for s in samples}),
            "distinctPixelHashes": len({s["imageSha256"] for s in samples}),
            "distinctPngHashes": len({s["pngSha256"] for s in samples}),
            "pngBytes": sum((EV / s["image"]).stat().st_size for s in samples),
            "timingsMedianUs": {
                key: round(stats.median([s["timingsUs"][key] for s in samples]), 1)
                for key in ("wait", "blit", "state", "digest", "pixels", "total")
            },
            "bulletsMin": min(s["activeBullets"] for s in samples),
            "bulletsMax": max(s["activeBullets"] for s in samples),
            "checksFailed": sum(
                1 for s in samples for passed in s["checks"].values() if not passed
            ),
        },
        "legacy": {
            "reads": len(legacy),
            "ticketChanged": sum(1 for r in legacy if not r["ticket_unchanged"]),
            "bulletDigestMismatch": sum(1 for r in legacy if not r["bullet_digest_match"]),
            "enemyDigestMismatch": sum(1 for r in legacy if not r["enemy_digest_match"]),
            "playerMismatch": sum(1 for r in legacy if not r["player_match"]),
            "frameMismatch": sum(1 for r in legacy if not r["frame_match"]),
        },
        "practice": {
            "tasks": practice["phases"]["M2_M3_trials"]["tasksTotal"],
            "frames": sum(g["frames"] for g in practice["clockAfterTrials"]["sessions"]),
            "sessions": practice["sessions"],
            "hits": len(practice["interruptions"]),
            "hitsPerAttempt": dict(Counter(h["attempt"] for h in practice["interruptions"])),
            "hitsPerPhase": dict(Counter(h["phase"] for h in practice["interruptions"])),
            "boundaries": [
                {
                    "name": b["name"],
                    "boundary": b["boundary"],
                    "expected": b["expected"],
                    "frames": b["frames"],
                    "staticTail": b["staticTail"],
                    "attempt": b["attempt"],
                }
                for b in practice["phases"]["M4_boundaries"]
            ],
            "reversals": [
                {
                    "name": block["name"],
                    "trials": len(block["trials"]),
                    "carryFrames": [t["carry"]["oldCarryFrames"] for t in block["trials"]],
                    "carryPx": [t["carry"]["oldCarryPx"] for t in block["trials"]],
                }
                for block in practice["phases"]["M4_carry"]["reversals"]
            ],
            "axisSwitches": [
                {
                    "name": block["name"],
                    "trials": len(block["trials"]),
                    "deltas": [t["carryOldAxis"]["deltas"] for t in block["trials"]],
                }
                for block in practice["phases"]["M4_carry"]["axisSwitches"]
            ],
        },
    }

    # Release behaviour: the corollary of "mask changes one frame before motion".
    release_stop = Counter()
    for class_trials in trials_raw.values():
        for trial in class_trials:
            release = trial.get("release") or []
            if len(release) < 2:
                release_stop["missing"] += 1
                continue
            movement = [
                max(abs(release[i]["x"] - release[i - 1]["x"]),
                    abs(release[i]["y"] - release[i - 1]["y"]))
                for i in range(1, len(release))
            ]
            stopped_at = next((i + 1 for i, value in enumerate(movement) if value == 0.0), None)
            release_stop[str(stopped_at)] += 1
    derived["practice"]["releaseStopFrame"] = dict(release_stop)

    (EV / "derived.json").write_text(json.dumps(derived, indent=2), encoding="utf-8")
    print(json.dumps(derived, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
