"""Dev-set evaluation and gate arithmetic (D7, first slice).

What this module is and is not
------------------------------
It runs the **subject** (controller, D5) and the **old baseline** (``AvoidPilot``,
frozen in D0) on the same synthetic instances and computes the gate arithmetic of
the acceptance document.  It is deliberately explicit about what it does *not*
have yet:

* the reference solver cannot label the long scenarios (it returns
  ``budget_exhausted`` on SC-15/16), so A-02/A-03 are computed on the subset where
  a label exists and marked ``subset``;
* the synthetic world has no bombs, so both arms run with ``bombs=0`` and no bomb
  effect is modelled — A-04 is a death-rate comparison, not a resource comparison;
* A-05/A-06 need per-family labels that D7 has not produced yet and are reported
  as ``not_computed`` rather than quietly passed.

Every gate carries its denominator and its status.  ``underpowered`` is a
first-class outcome: a baseline with zero deaths cannot support a "30% reduction".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import controller, families, forecast, planner, synth
from ..pilot import AvoidPilot, PilotConfig

GATE_SCHEMA = "v370-dev-eval/1"


@dataclass
class ArmResult:
    scenario_id: str
    arm: str
    survived: bool
    hit: bool
    hit_frame: int | None
    frames: int
    min_clearance: float | None
    entered_pocket: bool = False
    evacuate_frame: int | None = None
    pocket_frames: int = 0
    final_status: str | None = None
    statuses: dict | None = None
    final_reason: str | None = None


def _obs_for_pilot(x: float, y: float) -> dict:
    feat = np.zeros(24, dtype=np.float32)
    feat[0] = (x - 32.0) / 384.0
    feat[1] = (y - 16.0) / 448.0
    return {"self_feat": feat, "bullets": np.zeros((64, 6), dtype=np.float16), "danger": np.zeros((16, 8), dtype=np.uint8)}


def run_baseline(scenario: synth.Scenario, *, max_decisions: int | None = None) -> ArmResult:
    """Old pilot arm: 20-frame first-action search, no route, no forecast."""

    env = synth.SyntheticEnv(scenario)
    pilot = AvoidPilot(PilotConfig())
    limit = max_decisions or scenario.horizon_frames // env.scenario.frameskip
    decision = 0
    while not env.hit and env.frame < scenario.horizon_frames and decision < limit:
        mask = scenario.active(env.frame)
        positions = scenario.positions(env.frame)[mask]
        velocities = scenario.velocities()[mask]
        halves = scenario.halves()[mask]
        raw = {
            "position": positions.astype(np.float32),
            "velocity": velocities.astype(np.float32),
            # env.py collapses the box to one half extent for the old pilot.
            "half": np.max(halves, axis=1).astype(np.float32) if positions.size else np.zeros(0, np.float32),
        }
        action = pilot.act(_obs_for_pilot(env.x, env.y), raw, bombs=0)
        env.step((int(action[2]), int(action[0])))
        decision += 1
    return ArmResult(
        scenario_id=scenario.scenario_id,
        arm="baseline",
        survived=(not env.hit) and env.frame >= scenario.horizon_frames,
        hit=env.hit,
        hit_frame=env.frame if env.hit else None,
        frames=env.frame,
        min_clearance=None if env.min_clearance == float("inf") else round(env.min_clearance, 4),
    )


def run_subject(
    scenario: synth.Scenario,
    *,
    config: controller.ControllerConfig | None = None,
    max_decisions: int | None = None,
) -> ArmResult:
    config = config or controller.ControllerConfig()
    outcome = controller.run_episode(scenario, config=config, max_decisions=max_decisions)
    evacuate = next(
        (record["frame"] for record in outcome["decisions"] if record["intent"] in ("evacuate", "prepare_exit")),
        None,
    )
    # SC-15's dead-end pocket: 285 < x < 340 and 160 < y < 250.  Any frame there
    # means the route entered the cul-de-sac; the count is the A-07 evidence.
    pocket_frames = sum(
        1
        for record in outcome["decisions"]
        if 285.0 < record.get("x", -1e9) < 340.0 and 160.0 < record.get("y", -1e9) < 250.0
    )
    statuses: dict[str, int] = {}
    for record in outcome["decisions"]:
        status = record.get("plan_status") or "none"
        statuses[status] = statuses.get(status, 0) + 1
    final_status = outcome["decisions"][-1].get("plan_status") if outcome["decisions"] else None
    final_reason = outcome["decisions"][-1].get("reason") if outcome["decisions"] else None
    return ArmResult(
        scenario_id=scenario.scenario_id,
        arm="subject",
        survived=outcome["survived"],
        hit=outcome["hit"],
        hit_frame=outcome["frames"] if outcome["hit"] else None,
        frames=outcome["frames"],
        min_clearance=outcome["min_clearance"],
        evacuate_frame=evacuate,
        pocket_frames=pocket_frames,
        final_status=final_status,
        statuses=statuses,
        final_reason=final_reason,
    )


@dataclass
class Row:
    scenario_id: str
    family: str
    expected: str | None
    reference: str | None
    horizon: int
    subject: dict
    baseline: dict


def label_safe(payload) -> tuple[str, int | None]:
    """Last-safe-start label for one scenario; module level for worker pickling."""

    scenario, = payload
    from . import reference

    return scenario.scenario_id, reference.latest_safe_start(scenario)["latest_safe_start"]


def label_one(payload) -> tuple[str, str]:
    """Reference label for one scenario; module-level so workers can pickle it."""

    scenario, budget = payload
    from . import reference

    return scenario.scenario_id, reference.solve(scenario, node_budget=budget).status.value


def _row_for(
    scenario: synth.Scenario,
    subject_config: controller.ControllerConfig | None,
    include_baseline: bool,
    reference_status: dict[str, str] | None,
) -> dict:
    """One scenario, both arms.  Module level so it can run in a worker process."""

    subject = run_subject(scenario, config=subject_config)
    base = run_baseline(scenario) if include_baseline else None
    return asdict(
        Row(
            scenario_id=scenario.scenario_id,
            family=scenario.family,
            expected=scenario.expected,
            reference=(reference_status or {}).get(scenario.scenario_id),
            horizon=scenario.horizon_frames,
            subject=asdict(subject),
            baseline=asdict(base) if base is not None else None,
        )
    )


def evaluate(
    scenarios: list[synth.Scenario],
    *,
    subject_config: controller.ControllerConfig | None = None,
    include_baseline: bool = True,
    reference_status: dict[str, str] | None = None,
    workers: int = 1,
) -> dict:
    """Rows for every scenario.

    ``workers > 1`` fans the scenarios out over processes: the subject costs
    1-8 s per scenario and each one is independent, so this is close to linear on
    a multi-core host (measured: 21 scenarios, 8 workers, ~4x).
    """

    if workers <= 1:
        rows = [
            _row_for(scenario, subject_config, include_baseline, reference_status)
            for scenario in scenarios
        ]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_row_for, scenario, subject_config, include_baseline, reference_status)
                for scenario in scenarios
            ]
            rows = [future.result() for future in futures]
    return {"schema": GATE_SCHEMA, "rows": rows}


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def _status(value: bool | None) -> str:
    if value is None:
        return "not_computed"
    return "pass" if value else "fail"


def gates(document: dict, *, sparse_success: float | None = None, labels: dict | None = None) -> dict:
    rows = document["rows"]
    by_id = {row["scenario_id"]: row for row in rows}
    subject = lambda name: by_id.get(name, {}).get("subject", {})
    baseline = lambda name: (by_id.get(name) or {}).get("baseline") or {}
    # A-01: deterministic boundary instances.
    checks = {
        "E-01 crossing avoided": subject("E-01").get("hit") is False,
        "E-02 diagonal avoided": subject("E-02").get("hit") is False,
        "E-05 wide gap passable": subject("E-05").get("survived") is True,
        "E-06 narrow gap not passed": subject("E-06").get("survived") is False,
    }
    a01 = all(checks.values()) and "E-01" in by_id

    # A-02: success among instances the reference labelled feasible.
    eligible = [
        row
        for row in rows
        if row.get("reference") == "feasible"
        or (row.get("reference") is None and row["expected"] == "feasible")
    ]
    successes = sum(1 for row in eligible if row["subject"]["survived"])
    a02_rate = successes / len(eligible) if eligible else None

    # A-04: avoidable-death proxy — *relative* death reduction on paired instances.
    paired = [row for row in rows if row["baseline"] is not None]
    subject_deaths = sum(1 for row in paired if row["subject"]["hit"])
    baseline_deaths = sum(1 for row in paired if row["baseline"]["hit"])
    a04_reduction = (
        None
        if not paired or baseline_deaths == 0
        else round((baseline_deaths - subject_deaths) / baseline_deaths, 4)
    )
    a04_status = (
        "underpowered"
        if baseline_deaths == 0
        else _status(a04_reduction is not None and a04_reduction >= 0.3)
    )

    # A-05: instances labelled "must not pass" (no legal route) must not survive.
    must_not_pass = [
        row for row in rows
        if row["expected"] in ("infeasible_unproven", "blocked", "blocked_late")
    ]
    a05_pass = [row for row in must_not_pass if row["subject"]["survived"]]
    a05 = None if not must_not_pass else not a05_pass

    # A-06: status separation on the boundary instances.
    #   * a "blocked" instance must be certified, not merely timed out;
    #   * an "unproven/unknown" instance may only be certified when the record
    #     carries a proof marker (a late "18/18 first actions collide" is a proof,
    #     a bare "no route found" is not).
    proof_markers = ("coverage_proof", "all_first_actions_collide", "initial_overlap", "area_covered")

    def certified_with_proof(row) -> bool:
        subject_row = row["subject"]
        if subject_row.get("final_status") != "no_solution_certified":
            return False
        reason = subject_row.get("final_reason") or ""
        return any(marker in reason for marker in proof_markers)

    blocked_rows = [row for row in rows if row["expected"] == "blocked"]
    unknown_rows = [row for row in rows if row["expected"] in ("unknown", "infeasible_unproven")]
    # A-06b: a cover that appears *after* the plan horizon must be reported as
    # unknown/budget rather than silently treated as safe or certified.
    late_rows = [row for row in rows if row["expected"] == "blocked_late"]
    a06_late = None if not late_rows else all(
        row["subject"].get("final_status") in ("model_unknown", "budget_exhausted")
        or row["subject"].get("survived") is False
        for row in late_rows
    )
    a06_blocked = None if not blocked_rows else all(certified_with_proof(row) for row in blocked_rows)
    a06_unproven = None if not unknown_rows else all(
        row["subject"].get("final_status") != "no_solution_certified" or certified_with_proof(row)
        for row in unknown_rows
    )
    a06 = None if a06_blocked is None and a06_unproven is None and a06_late is None else bool(
        (a06_blocked is not False) and (a06_unproven is not False) and (a06_late is not False)
    )

    # A-03: on-time evacuation against the reference "last safe start" label.
    labelled = []
    for row in rows:
        latest = (labels or {}).get(row["scenario_id"])
        if latest is None:
            continue
        # Only closing instances (label before the horizon) that are solvable.
        if latest >= row.get("horizon", latest):
            continue
        if row.get("expected") == "feasible" or row.get("reference") == "feasible":
            labelled.append((row, latest))
    checked: list[bool] = []
    for row, latest in labelled:
        evacuate = row["subject"].get("evacuate_frame")
        if latest is None or evacuate is None:
            checked.append(False)
            continue
        checked.append(evacuate <= int(latest) - 10 and row["subject"]["survived"])
    a03 = None if not labelled else all(checked)
    a03_status = "underpowered" if not labelled else _status(a03)

    family_rates = per_family_success(document)
    weak_families = {
        name: entry["rate"]
        for name, entry in family_rates.items()
        if entry["counted"] and (entry["rate"] is None or entry["rate"] < 0.8)
    }
    a02 = (
        None
        if a02_rate is None
        else (a02_rate >= 0.9 and not weak_families)
    )

    # A-07: dead-end entry, aggregated over every SC-15 instance in the set.
    sc15_rows = [row for row in rows if row["family"] == "SC-15"]
    pocket = None
    if sc15_rows:
        pocket = max(int(row["subject"].get("pocket_frames") or 0) for row in sc15_rows)
    a07 = None if not sc15_rows else pocket == 0

    a08 = None if sparse_success is None else sparse_success >= 0.95

    return {
        "A-01": {
            "status": _status(a01),
            "checks": checks,
            "note": "deterministic boundary instances only; full matrix is D7 formal",
        },
        "A-02": {
            "status": _status(a02),
            "rate": None if a02_rate is None else round(a02_rate, 4),
            "denominator": len(eligible),
            "subset": True,
            "per_family": family_rates,
            "weak_families": weak_families,
            "note": (
                "reference-labelled subset only; each family with ≥3 feasible rows must "
                "reach ≥0.8 (acceptance A-02)"
            ),
        },
        "A-03": {
            "status": a03_status,
            "denominator": len(labelled),
            "labels": {
                scenario_id: latest for scenario_id, latest in (labels or {}).items()
            },
            "margin_frames": 10,
            "note": (
                "backward reachability label (8 px, cell-inflated) capped by the do-nothing "
                "death frame; evacuation must start 10 frames before it"
            ),
        },
        "A-04": {
            "status": a04_status,
            "baseline_deaths": baseline_deaths,
            "subject_deaths": subject_deaths,
            "pairs": len(paired),
            "relative_reduction": a04_reduction,
            "threshold": 0.3,
            "note": "relative death reduction on paired instances; no bombs in the synthetic world",
        },
        "A-05": {
            "status": _status(a05),
            "denominator": len(must_not_pass),
            "survived_anyway": [row["scenario_id"] for row in a05_pass],
            "note": "instances labelled infeasible/blocked must not survive",
        },
        "A-06": {
            "status": _status(a06),
            "blocked_instance_certified": a06_blocked,
            "unproven_not_certified": a06_unproven,
            "late_cover_reported_unknown": a06_late,
            "blocked_statuses": {
                row["scenario_id"]: row["subject"].get("final_status") for row in blocked_rows
            },
            "late_statuses": {
                row["scenario_id"]: row["subject"].get("final_status") for row in late_rows
            },
            "note": (
                "blocked needs a proof; unknown/unproven must never be reported as certified; "
                "a cover that appears beyond the plan horizon must be unknown/budget"
            ),
        },
        "A-07": {
            "status": _status(a07),
            "pocket_frames": pocket,
            "note": "entered the SC-15 dead end",
        },
        "A-08": {
            "status": _status(a08),
            "rate": None if sparse_success is None else round(sparse_success, 4),
            "note": "SC-17 sparse regression; small seed sample",
        },
    }


def per_family_success(document: dict, *, minimum_rows: int = 3) -> dict:
    """A-02 per family: success among *reference-feasible* rows only.

    The acceptance asks for "each solvable family ≥80%" with a visible denominator.
    Rows that the reference labelled unknown/budget are excluded — counting a
    deliberately infeasible instance against a family would make the rate depend
    on how many negatives happen to sit in that family.
    """

    out: dict[str, dict] = {}
    for row in document["rows"]:
        feasible = row.get("reference") == "feasible" or (
            row.get("reference") is None and row.get("expected") == "feasible"
        )
        if not feasible:
            continue
        name = row["family"]
        entry = out.setdefault(name, {"feasible": 0, "survived": 0})
        entry["feasible"] += 1
        entry["survived"] += 1 if row["subject"]["survived"] else 0
    for entry in out.values():
        entry["rate"] = round(entry["survived"] / entry["feasible"], 4) if entry["feasible"] else None
        entry["counted"] = entry["feasible"] >= minimum_rows
    return dict(sorted(out.items()))


def family_breakdown(document: dict) -> dict:
    """Per-family counts, so A-02's "each family ≥80%" has a visible denominator."""

    families: dict[str, dict] = {}
    for row in document["rows"]:
        name = row["family"]
        entry = families.setdefault(
            name,
            {
                "rows": 0,
                "subject_survived": 0,
                "baseline_survived": 0,
                "reference_feasible": 0,
                "expected": {},
            },
        )
        entry["rows"] += 1
        entry["subject_survived"] += 1 if row["subject"]["survived"] else 0
        if row.get("baseline"):
            entry["baseline_survived"] += 1 if row["baseline"]["survived"] else 0
        if row.get("reference") == "feasible" or (
            row.get("reference") is None and row.get("expected") == "feasible"
        ):
            entry["reference_feasible"] += 1
        label = row.get("expected") or "none"
        entry["expected"][label] = entry["expected"].get(label, 0) + 1
    for entry in families.values():
        entry["subject_rate"] = (
            round(entry["subject_survived"] / entry["rows"], 4) if entry["rows"] else None
        )
        entry["baseline_rate"] = (
            round(entry["baseline_survived"] / entry["rows"], 4) if entry["rows"] else None
        )
    return dict(sorted(families.items()))


def settle(document: dict, *, labels: dict | None = None, sparse_success: float | None = None) -> dict:
    """Machine-readable settlement: gates + per-family denominators.

    The markdown settlement page quotes these numbers; writing them here keeps the
    prose checkable instead of hand-copied.
    """

    if sparse_success is None:
        sc17 = [row for row in document["rows"] if row["family"] == "SC-17"]
        if sc17:
            sparse_success = sum(1 for row in sc17 if row["subject"]["survived"]) / len(sc17)
    gate_table = gates(document, sparse_success=sparse_success, labels=labels)
    return {
        "schema": "v370-settlement/1",
        "split": document.get("split", "dev"),
        "scenarios": len(document["rows"]),
        "gates": gate_table,
        "family_breakdown": family_breakdown(document),
        "reference_labels": dict(sorted((labels or {}).items())),
    }


def render_markdown(settlement: dict) -> str:
    """Markdown tables generated from a settlement document.

    The prose settlement page quotes these numbers; generating the tables from the
    data removes the hand-copy step that drifted once already (D28b).
    """

    lines = [
        f"### 门禁（{settlement.get('split', 'dev')}，{settlement['scenarios']} 个场景）",
        "",
        "| 门 | 状态 | 关键数值 |",
        "|---|---|---|",
    ]
    for gate, payload in settlement["gates"].items():
        parts = []
        for key in ("rate", "relative_reduction", "denominator", "pairs", "mean_rate"):
            if key in payload and payload[key] is not None:
                parts.append(f"{key}={payload[key]}")
        if gate == "A-05" and payload.get("survived_anyway"):
            parts.append(f"survived_anyway={payload['survived_anyway']}")
        lines.append(f"| {gate} | {payload['status']} | {'; '.join(parts) or '—'} |")

    lines += ["", "| 族 | 行数 | 新方案 | baseline | 参考可解 |", "|---|---:|---:|---:|---:|"]
    for name, entry in settlement["family_breakdown"].items():
        lines.append(
            f"| {name} | {entry['rows']} | {entry['subject_rate']} | {entry['baseline_rate']} | "
            f"{entry['reference_feasible']} |"
        )

    a02 = settlement["gates"].get("A-02") or {}
    per_family = a02.get("per_family") or {}
    if per_family:
        lines += ["", "| 族（A-02） | 存活/可行 | 比率 | 计入判定 |", "|---|---:|---:|---|"]
        for name, entry in per_family.items():
            lines.append(
                f"| {name} | {entry['survived']}/{entry['feasible']} | {entry['rate']} | "
                f"{entry['counted']} |"
            )
    return "\n".join(lines) + "\n"


def write(document: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
