"""Frozen old-pilot baseline and a digest-pinned golden fixture (D0).

Why a golden fixture instead of "we did not touch pilot.py"
-----------------------------------------------------------
The v3.7.0 plan keeps the old controller as the *baseline arm*.  The working
tree already carries uncommitted edits, so ``git diff`` alone cannot prove that
the baseline used for a later comparison is the one that was audited here.  The
fixture pins three things:

* the exact :class:`~touhou_rl.pilot.PilotConfig` values used as the baseline,
* the SHA-256 of ``touhou_rl/pilot.py`` that produced the recorded decisions,
* the 18-way decision values and the chosen action on a fixed set of synthetic
  fields, including the two failure modes the new version exists to fix
  (a wall that is outside the 240 px near radius, and a crossing that happens
  between two ``frameskip`` samples).

Regenerating is intentionally a no-op when nothing changed; a changed decision
digest means the baseline moved and every later paired comparison must say so.

The cases use absolute playfield coordinates (px) and per-frame velocities, so
they exercise the same ``bullets_raw`` path the live probe uses.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import spec
from ..pilot import AvoidPilot, PilotConfig
from . import geometry
from .contracts import canonical_json

SCHEMA = "v370-baseline-golden/1"

#: The audited baseline.  Only fields that change behaviour are listed; a
#: missing key means "the dataclass default at fixture time", which is recorded
#: verbatim in the generated fixture so the default cannot drift silently.
BASELINE_OVERRIDES: dict[str, float | int | bool] = {}

#: Fields whose value is part of the baseline's identity.
BASELINE_FIELDS = (
    "frameskip",
    "horizon_steps",
    "beam",
    "quota_per_action",
    "speed",
    "speed_diag",
    "focus_speed",
    "focus_speed_diag",
    "player_half",
    "bullet_pad",
    "near_radius",
    "max_bullets",
    "bottom_bias",
    "aim_weight",
    "aim_safe_margin",
    "switch_penalty",
    "bomb_min_clearance",
    "bomb_cooldown",
    "always_shoot",
)


def legacy_config() -> PilotConfig:
    return PilotConfig(**BASELINE_OVERRIDES)


def config_snapshot() -> dict:
    config = legacy_config()
    return {name: getattr(config, name) for name in BASELINE_FIELDS}


@dataclass(frozen=True)
class GoldenCase:
    """One synthetic field plus the reason it is pinned."""

    name: str
    purpose: str
    player: tuple[float, float]
    hazards: tuple[tuple[float, float, float, float, float, float], ...] = ()
    bombs: int = 0
    bosses: tuple[tuple[float, float], ...] = ()
    #: Optional independent reference (not produced by the pilot).
    analytic: dict | None = None


def _hazard(x, y, vx, vy, half=2.0) -> tuple[float, float, float, float, float, float]:
    return (float(x), float(y), float(vx), float(vy), float(half), float(half))


def _ring(count: int, cx: float, cy: float, radius: float) -> tuple:
    out = []
    for index in range(count):
        angle = 2.0 * np.pi * index / count
        out.append(_hazard(cx + radius * np.cos(angle), cy + radius * np.sin(angle), 0.0, 0.0))
    return tuple(out)


CASES: tuple[GoldenCase, ...] = (
    GoldenCase(
        name="empty-field",
        purpose="empty pool is not an error and must not spend a bomb",
        player=(192.0, 384.0),
        bombs=3,
    ),
    GoldenCase(
        name="head-on-above",
        purpose="a falling bullet must not be met by moving up (basic sanity)",
        player=(192.0, 384.0),
        hazards=(_hazard(192.0, 336.0, 0.0, 3.0), _hazard(202.0, 324.0, 0.0, 3.0)),
        bombs=3,
    ),
    GoldenCase(
        name="static-wall-left",
        purpose="a wall closing the left side must push the baseline right",
        player=(192.0, 384.0),
        hazards=tuple(_hazard(150.0, y, 0.0, 0.0) for y in range(300, 470, 12)),
        bombs=3,
    ),
    GoldenCase(
        name="far-static-wall",
        purpose=(
            "FR-05 / H1: a static wall 300 px away closes the exit in ~75 frames but is "
            "outside near_radius=240, so the 20-frame baseline plans as if the field were empty"
        ),
        player=(192.0, 384.0),
        hazards=tuple(_hazard(x, 84.0, 0.0, 0.0) for x in range(40, 410, 16)),
        bombs=3,
    ),
    GoldenCase(
        name="pool-truncation",
        purpose=(
            "FR-05: with 400 active bullets the baseline keeps the 320 nearest and silently "
            "drops the rest; bullets_used records the truncation instead of claiming coverage"
        ),
        player=(192.0, 384.0),
        hazards=_ring(400, 192.0, 384.0, 200.0),
        bombs=0,
    ),
    GoldenCase(
        name="dense-unavoidable-with-bomb",
        purpose="a filled neighbourhood must spend a bomb when one is available",
        player=(192.0, 384.0),
        hazards=tuple(
            _hazard(x, y, 0.0, 0.0) for x in range(183, 202, 2) for y in range(375, 394, 2)
        ),
        bombs=2,
    ),
    GoldenCase(
        name="dense-unavoidable-no-bomb",
        purpose="the same field without a bomb must not report a bomb action",
        player=(192.0, 384.0),
        hazards=tuple(
            _hazard(x, y, 0.0, 0.0) for x in range(183, 202, 2) for y in range(375, 394, 2)
        ),
        bombs=0,
    ),
    GoldenCase(
        name="crossing-between-samples",
        purpose=(
            "SC-08: the bullet is clear at every 2-frame sample (t=2, 4, ...) but overlaps the "
            "hurtbox at t=3, so per-step sampling cannot see it while the swept test can"
        ),
        player=(192.0, 384.0),
        hazards=(_hazard(163.0, 384.0, 10.0, 0.0),),
        bombs=0,
        analytic={
            "model": "stationary player, 20 frames, box-vs-box swept clearance",
            "path": "player holds position at (192, 384)",
            "frames": 20,
        },
    ),
    GoldenCase(
        name="mirror-right",
        purpose="mirror pair: the same geometry with the opening on the right",
        player=(192.0, 384.0),
        hazards=tuple(_hazard(230.0 + 8.0 * i, 320.0 - 6.0 * i, -1.0, 2.0) for i in range(6)),
        bombs=2,
    ),
    GoldenCase(
        name="mirror-left",
        purpose=(
            "mirror pair: the x-reflected field (x' = 448 - x, player' = 448 - player) must "
            "produce the x-reflected decision, otherwise the baseline is remembering a side "
            "instead of the geometry"
        ),
        player=(256.0, 384.0),
        hazards=tuple(_hazard(448.0 - (230.0 + 8.0 * i), 320.0 - 6.0 * i, 1.0, 2.0) for i in range(6)),
        bombs=2,
    ),
    GoldenCase(
        name="corner-clamp",
        purpose=(
            "SC-06/SC-14: at the right wall the baseline clips candidate positions 0.825 px "
            "inside the real clamp, so its plan is slightly conservative rather than illegal"
        ),
        player=(376.0, 384.0),
        hazards=tuple(_hazard(376.0, y, 0.0, 0.0) for y in range(250, 300, 10)),
        bombs=0,
    ),
)


def _observation(player_xy: tuple[float, float]) -> dict:
    feat = np.zeros(24, dtype=np.float32)
    feat[0] = (player_xy[0] - spec.PLAYFIELD_X) / spec.PLAYFIELD_W
    feat[1] = (player_xy[1] - spec.PLAYFIELD_Y) / spec.PLAYFIELD_H
    return {
        "self_feat": feat,
        "bullets": np.zeros((64, 6), dtype=np.float16),
        "danger": np.zeros((16, 8), dtype=np.uint8),
    }


def _raw(case: GoldenCase) -> dict:
    rows = np.asarray(case.hazards, dtype=np.float32).reshape(-1, 6)
    return {
        "position": rows[:, :2].copy(),
        "velocity": rows[:, 2:4].copy(),
        # ``pilot._bullets_from_raw`` collapses the box to one half extent; the
        # fixture feeds max(half_x, half_y), which is what the live env exposes.
        "half": np.maximum(rows[:, 4], rows[:, 5]) if rows.size else np.zeros(0, dtype=np.float32),
    }


def _bosses(case: GoldenCase):
    return np.asarray(case.bosses, dtype=np.float32).reshape(-1, 2) if case.bosses else None


def run_case(case: GoldenCase) -> dict:
    """Return the baseline's decision values and action for one case (no mutation)."""
    observation = _observation(case.player)
    raw = _raw(case)
    valuator = AvoidPilot(legacy_config())
    values = valuator.candidate_scores(observation, raw, bosses=_bosses(case))
    actor = AvoidPilot(legacy_config())
    action = actor.act(observation, raw, bombs=case.bombs, bosses=_bosses(case))
    record: dict = {
        "name": case.name,
        "purpose": case.purpose,
        "action": [int(value) for value in action],
        "clearance": round(float(actor.debug.best_clearance), 4),
        "bullets_used": int(actor.debug.bullets_used),
        "hazards_supplied": len(case.hazards),
        "bombs_available": case.bombs,
        "values": [round(float(value), 4) for value in values],
    }
    if case.analytic is not None:
        path = np.repeat(np.asarray([case.player], dtype=np.float64), case.analytic["frames"] + 1, axis=0)
        single = case.hazards[0]
        clearance, segment, _hazard = geometry.path_min_clearance(
            path,
            np.asarray([single[:2]]),
            np.asarray([single[2:4]]),
            np.asarray([single[4:6]]),
            geometry.MoveModel().hurtbox_half,
        )
        # What the baseline could see: the same field sampled every frameskip.
        steps = np.arange(0, case.analytic["frames"] + 1, legacy_config().frameskip, dtype=np.float64)
        sampled = np.min(
            np.abs(case.player[0] - (single[0] + single[2] * steps)) - (single[4] + geometry.MoveModel().hurtbox_half)
        )
        record["analytic"] = {
            **case.analytic,
            "swept_min_clearance": round(clearance, 4),
            "swept_first_segment": int(segment),
            "sample_grid_min_clearance_x_axis": round(float(sampled), 4),
        }
    return record


def _pilot_sha256() -> str:
    path = Path(__file__).resolve().parents[1] / "pilot.py"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(records: list[dict], keys=("name", "action")) -> str:
    payload = [{key: record[key] for key in keys} for record in records]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def golden_document() -> dict:
    records = [run_case(case) for case in CASES]
    return {
        "schema": SCHEMA,
        "generated_by": "touhou_rl.escape.baseline",
        "generator": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "baseline": {
            "config": config_snapshot(),
            "overrides": BASELINE_OVERRIDES,
            "pilot_sha256": _pilot_sha256(),
        },
        "cases": records,
        "digests": {
            "decisions": _digest(records),
            "values": _digest(records, keys=("name", "values")),
            "cases": hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest(),
        },
    }


def write_golden(path: Path) -> dict:
    document = golden_document()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return document


def verify_golden(path: Path) -> list[str]:
    """Compare a committed fixture with a fresh run; return human-readable diffs."""
    if not path.exists():
        return [f"missing fixture: {path}"]
    committed = json.loads(path.read_text(encoding="utf-8"))
    fresh = golden_document()
    issues: list[str] = []
    if committed.get("schema") != SCHEMA:
        issues.append(f"schema: {committed.get('schema')!r} != {SCHEMA!r}")
    for key in ("pilot_sha256",):
        if committed.get("baseline", {}).get(key) != fresh["baseline"][key]:
            issues.append(f"baseline.{key}: {committed.get('baseline', {}).get(key)} != {fresh['baseline'][key]}")
    for key in ("decisions", "values", "cases"):
        if committed.get("digests", {}).get(key) != fresh["digests"][key]:
            issues.append(f"digests.{key}: committed {committed.get('digests', {}).get(key)} != fresh {fresh['digests'][key]}")
    committed_cases = {record["name"]: record for record in committed.get("cases", [])}
    for record in fresh["cases"]:
        old = committed_cases.get(record["name"])
        if old is None:
            issues.append(f"case {record['name']}: missing from fixture")
            continue
        if old.get("action") != record["action"]:
            issues.append(f"case {record['name']}: action {old.get('action')} -> {record['action']}")
        if old.get("bullets_used") != record["bullets_used"]:
            issues.append(f"case {record['name']}: bullets_used {old.get('bullets_used')} -> {record['bullets_used']}")
    return issues
