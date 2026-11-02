"""Round28: can any existing teacher produce labels for real pixels?

P3 needs labels that a *real* session can produce.  The synthetic teachers were
built on worlds that carry parameters the real game does not expose (``gap``,
``coverage``, ``wall_frame``, ``window``, ``threshold``, ``goal`` ...).  This tool
measures exactly which ones are needed:

* **baseline** – teacher on the real world (reproduces round19's 14/28 and 28/28);
* **masked** – teacher on a proxy that raises ``PrivilegeBlocked`` when the
  *teacher module itself* reads a privileged field (the world's own dynamics are
  still forwarded, so a rollout inside the teacher is not misreported as a
  privileged read);
* **corrupted** – teacher that believes wrong parameters while the world advances
  with the real ones: if success survives, the parameter was not load-bearing.

The question is not "is the teacher good" but "could this label exist in a real
session at all".  F1's wall/gap is not rendered before the lethal frame in the
original renderer (round20) and only the research telegraph shows it (round21),
so a task that needs ``gap`` cannot have a state-only label source.

    python tools/rl/v373_privilege_audit.py --out Document/Update/Alpha/v3.7.3/evidence/privilege-audit-round28
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVIDENCE = ROOT / "Document/Update/Alpha/v3.7.3/evidence/worldfilter-round18/run/filters"
FAMILIES = ("F1", "F2", "F3")

# Fields a real session cannot hand to a label generator.
PRIVILEGED = frozenset(
    {
        "coverage",
        "gap",
        "internal_gap",
        "wall_frame",
        "window",
        "threshold",
        "goal",
        "origin",
        "burst",
        "fire_interval",
        "bullet_speed",
        "start_x",
        "depletion_pressure",
    }
)


class PrivilegeBlocked(RuntimeError):
    def __init__(self, field: str):
        super().__init__(f"privileged read: {field}")
        self.field = field


class MaskedWorld:
    """Forward the world, but block privileged reads made by the teacher itself."""

    def __init__(self, real, teacher_files: tuple[str, ...]):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_teacher_files", teacher_files)

    def __getattr__(self, name):
        if name in PRIVILEGED:
            caller = sys._getframe(1).f_code.co_filename
            if any(caller.endswith(suffix) for suffix in self._teacher_files):
                raise PrivilegeBlocked(name)
        return getattr(self._real, name)


def load_worlds() -> list[dict]:
    from touhou_rl import tactical_teacher_validation as tv

    worlds = []
    for family in FAMILIES:
        for path in sorted((EVIDENCE / family).glob(f"wf18-{family}-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            scenario = dict(payload["scenario"])
            worlds.append(
                {
                    "eventId": payload["event_id"],
                    "family": family,
                    "mirror": bool(payload["mirror"]),
                    "group": payload["group"],
                    "scenario": scenario,
                    "world": tv.ValidationWorld(**scenario),
                }
            )
    return worlds


def run_teacher(teacher, decision_world, real_world) -> dict:
    """Run the teacher's decisions against the *real* dynamics."""
    from touhou_rl import tactical_teacher_validation as tv
    from touhou_rl.tactical_scenarios import initial_state

    state = initial_state(real_world)
    actions = []
    blocked = None
    error = None
    try:
        while state.frame < real_world.horizon:
            decision = teacher.decide(decision_world, state)
            actions.append(int(decision.action))
            state = tv.advance(real_world, state, decision.action)
            if state.hit:
                break
    except PrivilegeBlocked as exc:
        blocked = exc.field
    except Exception as exc:  # noqa: BLE001 - a crashing teacher is a result
        error = f"{type(exc).__name__}: {exc}"
    verdict = None
    if blocked is None and error is None:
        try:
            verdict = tv.verify(real_world, tuple(actions))
        except Exception as exc:  # noqa: BLE001
            error = f"verify: {type(exc).__name__}: {exc}"
    return {
        "blockedField": blocked,
        "error": error,
        "verdict": verdict,
        "success": verdict == "feasible",
        "actions": actions[:1] if not actions else actions,
    }


def perturb(world, family: str):
    """A believable but wrong parameter set for the same world."""
    changes = {}
    if world.internal_gap if hasattr(world, "internal_gap") else False:
        gap = list(world.gap)
        changes["gap"] = (min(330.0, gap[0] + 60.0), min(360.0, gap[1] + 60.0))
    if world.family == "F1":
        changes["coverage"] = (world.coverage[0] + 24.0, world.coverage[1] - 24.0)
        changes["wall_frame"] = max(20, int(world.wall_frame) - 12)
    if world.family in ("F2", "F3"):
        window = list(world.window)
        changes["window"] = (max(0, window[0] - 18), window[1] + 18)
        changes["threshold"] = min(340.0, world.threshold + 28.0)
        changes["goal"] = max(20.0, world.goal - 30.0)
        changes["fire_interval"] = max(6, int(world.fire_interval) + 4)
        changes["bullet_speed"] = max(2.0, world.bullet_speed + 2.0)
    changes["start_x"] = min(368.0, world.start_x + 30.0)
    return dataclasses.replace(world, **changes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("v373-privilege-audit")
    parser.add_argument(
        "--out",
        default="Document/Update/Alpha/v3.7.3/evidence/privilege-audit-round28",
    )
    args = parser.parse_args(argv)

    from touhou_rl.tactical_teacher import TacticalTeacher
    from touhou_rl.tactical_teacher_v2 import TacticalTeacherV2

    teachers = {
        "old": (TacticalTeacher(), ("tactical_teacher.py",)),
        "v2": (TacticalTeacherV2(), ("tactical_teacher_v2.py",)),
    }
    worlds = load_worlds()
    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "schema": "v373-privilege-audit/1",
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "worlds": len(worlds),
        "byFamily": {family: sum(1 for w in worlds if w["family"] == family) for family in FAMILIES},
        "privilegedFields": sorted(PRIVILEGED),
        "teachers": {},
    }

    for name, (teacher, files) in teachers.items():
        rows = []
        for item in worlds:
            real = item["world"]
            baseline = run_teacher(teacher, real, real)
            masked = run_teacher(teacher, MaskedWorld(real, files), real)
            corrupted_world = perturb(real, item["family"])
            corrupted = run_teacher(teacher, corrupted_world, real)
            rows.append(
                {
                    "eventId": item["eventId"],
                    "family": item["family"],
                    "mirror": item["mirror"],
                    "baseline": {"success": baseline["success"], "verdict": baseline["verdict"],
                                 "error": baseline["error"]},
                    "masked": {
                        "success": masked["success"],
                        "verdict": masked["verdict"],
                        "blockedField": masked["blockedField"],
                        "error": masked["error"],
                    },
                    "corrupted": {"success": corrupted["success"], "verdict": corrupted["verdict"],
                                  "error": corrupted["error"]},
                }
            )
        summary = {
            "baseline": sum(1 for row in rows if row["baseline"]["success"]),
            "maskedCompletes": sum(
                1 for row in rows if row["masked"]["blockedField"] is None and row["masked"]["error"] is None
            ),
            "maskedSuccess": sum(1 for row in rows if row["masked"]["success"]),
            "corruptedSuccess": sum(1 for row in rows if row["corrupted"]["success"]),
            "total": len(rows),
        }
        by_family = {}
        for family in FAMILIES:
            family_rows = [row for row in rows if row["family"] == family]
            blocked = sorted(
                {
                    row["masked"]["blockedField"]
                    for row in family_rows
                    if row["masked"]["blockedField"]
                }
            )
            by_family[family] = {
                "n": len(family_rows),
                "baseline": sum(1 for row in family_rows if row["baseline"]["success"]),
                "maskedCompletes": sum(
                    1
                    for row in family_rows
                    if row["masked"]["blockedField"] is None and row["masked"]["error"] is None
                ),
                "maskedSuccess": sum(1 for row in family_rows if row["masked"]["success"]),
                "corruptedSuccess": sum(1 for row in family_rows if row["corrupted"]["success"]),
                "blockedFields": blocked,
                "stateOnlyAdmissible": all(
                    row["masked"]["blockedField"] is None and row["masked"]["error"] is None
                    for row in family_rows
                ),
            }
        result["teachers"][name] = {"summary": summary, "byFamily": by_family, "rows": rows}

    # Verdict, in two layers so measurement and argument are not confused:
    #   * stateOnlyByInputs: measured — some teacher completes every world of the
    #     family without a privileged read.
    #   * realWorldTaskObjects: argued, not measured — do the objects the family
    #     is defined by exist in TH08 at all?  F1 is a synthetic coverage wall,
    #     F3 a synthetic depletion window; only F2's aimed bullets are real game
    #     objects (state.py reads the bullet pool and the emission cursors).
    # A label source for real pixels needs BOTH.
    task_objects = {"F1": False, "F2": True, "F3": False}
    admissible = {}
    for family in FAMILIES:
        per_teacher = {
            name: result["teachers"][name]["byFamily"][family]["stateOnlyAdmissible"]
            for name in teachers
        }
        by_inputs = any(per_teacher.values())
        admissible[family] = {
            "stateOnlyByInputs": by_inputs,
            "realWorldTaskObjects": task_objects[family],
            "labelSourceAdmissible": by_inputs and task_objects[family],
            "perTeacher": per_teacher,
        }
    result["verdict"] = {
        "stateOnlyByInputs": [f for f in FAMILIES if admissible[f]["stateOnlyByInputs"]],
        "realWorldTaskObjects": task_objects,
        "labelSourceAdmissible": [
            f for f in FAMILIES if admissible[f]["labelSourceAdmissible"]
        ],
        "perFamily": admissible,
        "note": (
            "F1's wall/gap is a world parameter that the original renderer never "
            "draws before the lethal frame (round20); only the research telegraph "
            "shows it (round21).  F3's depletion window is the same kind of "
            "synthetic parameter.  Only F2 (aimed bullets and their emissions) is "
            "both state-only measurable and a real TH08 object."
        ),
    }
    result["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"teachers": {k: v["summary"] for k, v in result["teachers"].items()},
                      "byFamily": {k: v["byFamily"] for k, v in result["teachers"].items()},
                      "verdict": result["verdict"]}, indent=2))
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
