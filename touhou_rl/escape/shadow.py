"""Shadow re-check of recorded decisions with the exact AABB sweep (D1).

What it answers
---------------
Two claims from D0 are testable against data already on disk:

* the old pilot's circle model is not the game's box-vs-box test, and
* its per-``frameskip`` sampling cannot see a crossing between samples.

Method
------
For every recorded decision the *executed* path is rebuilt with the measured
one-frame input settle (:func:`touhou_rl.escape.geometry.executed_path`): the
first frame after a key change still runs the previous key state.  That predicts
the next snapshot exactly in 96.6% of 2,903 pairs (P95 residual 0.001 px), versus
65.9% if the new velocity is assumed from frame 1.

The bullet pool recorded at the start of the interval is propagated linearly,
and the exact swept box clearance is computed over the executed path.  The
pilot's own residual (``teacher.clearance``) and a recomputation of the pilot's
model (new velocity from frame 1, circle radius, sampled at ``frameskip``) are
kept as references, not as truth.

Interpretation limits, stated up front:

* the sidecar stores one half extent per bullet (``max(half_x, half_y)``), so the
  shadow box is square with that extent and *over*-approximates anisotropic
  bullets; a reported collision is therefore a conservative claim;
* the phase of the pool within its frame is unknown to within one frame;
* ``teacher.clearance`` is the pilot's residual over its own 20-frame plan, so
  "reported safe but the sweep disagrees" is evidence of a model/geometry gap,
  not a physical no-hit proof.

Pairs where the player is dying or a death was recorded are skipped, because
positions freeze there.  Whether a collision lands just before a recorded death
is reported separately: it is a sanity check of the geometry against the game's
own event stream.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry
from .contracts import HURTBOX_HALF_PX

SCHEMA = "v370-shadow/1"

#: The old pilot's radius pad (``pilot.PilotConfig.bullet_pad``).
LEGACY_PAD = 0.5
#: The old pilot only samples the ends of each ``frameskip`` step.
LEGACY_FRAMESKIP = 2
#: A collision this many decisions before a recorded death still counts as
#: "the sweep saw the fatal bullets" (deaths can take a frame or two).
DEATH_WINDOW_DECISIONS = 3


@dataclass(frozen=True)
class ShadowEvent:
    decision: int
    frame: int
    frames: int
    frame_clearance: float
    frame_index: int
    swept_clearance: float
    pilot_model_clearance: float
    teacher_clearance: float | None


def load_sidecar(path: Path | str) -> dict:
    data = np.load(Path(path))
    return {
        "positions": np.asarray(data["positions"], dtype=np.float64),
        "velocities": np.asarray(data["velocities"], dtype=np.float64),
        "half": np.asarray(data["half"], dtype=np.float64),
        "offsets": np.asarray(data["offsets"], dtype=np.int64),
    }


def _pool(sidecar: dict, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = int(sidecar["offsets"][index])
    end = int(sidecar["offsets"][index + 1])
    return (
        sidecar["positions"][start:end],
        sidecar["velocities"][start:end],
        sidecar["half"][start:end],
    )


def _pilot_model_clearance(
    start: np.ndarray,
    velocity: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    half: np.ndarray,
    hurtbox: float,
    frames: int,
) -> float:
    """What the old model would report: new velocity from frame 1, circle, sampled at step ends."""
    box = geometry.MoveModel()
    radius = half + hurtbox + LEGACY_PAD
    best = np.inf
    position = start.copy()
    for step in range(1, frames + 1):
        position = np.array(box.clamp(*(position + velocity)))
        if step % LEGACY_FRAMESKIP:
            continue
        centres = positions + velocities * float(step)
        distance = np.sqrt(((position - centres) ** 2).sum(axis=1)) - radius
        if distance.size:
            best = min(best, float(distance.min()))
    return float(best) if np.isfinite(best) else 60.0


def shadow_episode(
    trace_path: Path | str,
    sidecar_path: Path | str,
    *,
    max_decisions: int | None = None,
    margin: float = 0.0,
    examples: int = 12,
    skip_overrides: bool = False,
) -> dict:
    trace_path = Path(trace_path)
    sidecar = load_sidecar(sidecar_path)
    records: list[dict] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_decisions is not None and len(records) >= max_decisions + 1:
                break

    death_decisions = [
        int(record.get("decision", index + 1))
        for index, record in enumerate(records)
        if "death" in (record.get("events") or [])
    ]

    checked = 0
    skipped: Counter[str] = Counter()
    clearances: list[float] = []
    phase_errors: list[float] = []
    reported_safe_crossing = 0
    frame_collisions = 0
    swept_collisions = 0
    swept_only = 0
    aligned = 0
    teacher_above = 0
    teacher_below = 0
    before_death = 0
    events: list[ShadowEvent] = []
    pool_sizes: list[int] = []

    for index in range(1, len(records)):
        # The decision recorded at ``current`` was computed from the snapshot
        # recorded at ``previous`` (probe35 passes ``prev_snapshot`` to the
        # pilot), and its action executes over that interval.
        previous = records[index - 1]
        current = records[index]
        frame0 = int(previous.get("frame_post", -1))
        frame1 = int(current.get("frame_post", -1))
        frames = frame1 - frame0
        if frame0 < 0 or frames <= 0 or frames > 6:
            skipped["frame_alignment"] += 1
            continue
        player0 = previous.get("player") or {}
        player1 = current.get("player") or {}
        if not {"x", "y", "vx", "vy"} <= set(player0):
            skipped["player_missing"] += 1
            continue
        if int(player0.get("state", 0)) == 2 or int(player1.get("state", 0)) == 2:
            skipped["dying"] += 1
            continue
        if int(player0.get("state", 0)) == 3 or int(player1.get("state", 0)) == 3:
            # Invulnerable frames pass bullets through; counting them as hits
            # would inflate the screen without saying anything about dodging.
            skipped["invulnerable"] += 1
            continue
        controller = previous.get("controller") or {}
        if skip_overrides and controller.get("override"):
            # A tactic replaced the pilot's action; ``teacher.clearance`` still
            # describes the pilot's own pick, so the comparison would be invalid.
            skipped["controller_override"] += 1
            continue
        deaths_before = (previous.get("resources") or {}).get("deaths")
        deaths_after = (current.get("resources") or {}).get("deaths")
        if isinstance(deaths_before, int) and isinstance(deaths_after, int) and deaths_after > deaths_before:
            skipped["death"] += 1
            continue
        hurtbox = max(
            float(player0.get("hurtbox_half_x", HURTBOX_HALF_PX)),
            float(player0.get("hurtbox_half_y", HURTBOX_HALF_PX)),
        )
        positions, velocities, bullet_half = _pool(sidecar, index - 1)
        trace_count = (previous.get("bullets") or {}).get("count")
        if isinstance(trace_count, int) and positions.shape[0] != trace_count:
            skipped["pool_size_mismatch"] += 1
            continue
        checked += 1
        pool_sizes.append(int(positions.shape[0]))

        start = np.array([player0["x"], player0["y"]], dtype=np.float64)
        velocity_previous = np.array([player0["vx"], player0["vy"]], dtype=np.float64)
        velocity_current = np.array(
            [player1.get("vx", velocity_previous[0]), player1.get("vy", velocity_previous[1])],
            dtype=np.float64,
        )
        path = geometry.executed_path(start, velocity_previous, velocity_current, frames)
        actual_end = np.array([player1["x"], player1["y"]], dtype=np.float64)
        phase_errors.append(float(np.linalg.norm(actual_end - path[-1])))

        if positions.shape[0] == 0:
            continue
        box_half = np.repeat(bullet_half[:, None], 2, axis=1)
        swept, swept_segment, _hazard = geometry.path_min_clearance(
            path, positions, velocities, box_half, hurtbox
        )
        frame_clear, frame_index, _hazard = geometry.frame_min_clearance(
            path, positions, velocities, box_half, hurtbox, first_frame=1
        )
        clearances.append(frame_clear)
        if swept <= margin:
            swept_collisions += 1
            if frame_clear > margin:
                swept_only += 1
        teacher = (current.get("teacher") or {}).get("clearance")
        teacher_value = float(teacher) if isinstance(teacher, (int, float)) else None
        if frame_clear <= margin:
            frame_collisions += 1
            if teacher_value is not None and teacher_value > margin:
                reported_safe_crossing += 1
            decision = int(current.get("decision", index + 1))
            if any(0 <= death - decision <= DEATH_WINDOW_DECISIONS for death in death_decisions):
                before_death += 1
            if len(events) < examples:
                pilot_clearance = _pilot_model_clearance(
                    start, velocity_current, positions, velocities, bullet_half, hurtbox, frames
                )
                if teacher_value is not None:
                    if abs(pilot_clearance - teacher_value) < 0.01:
                        aligned += 1
                    elif teacher_value > pilot_clearance:
                        teacher_above += 1
                    else:
                        teacher_below += 1
                events.append(
                    ShadowEvent(
                        decision=decision,
                        frame=frame0,
                        frames=frames,
                        frame_clearance=round(frame_clear, 4),
                        frame_index=int(frame_index),
                        swept_clearance=round(swept, 4),
                        pilot_model_clearance=round(pilot_clearance, 4),
                        teacher_clearance=None if teacher_value is None else round(teacher_value, 4),
                    )
                )

    values = np.asarray(clearances, dtype=np.float64) if clearances else np.zeros(0)
    phase = np.asarray(phase_errors, dtype=np.float64) if phase_errors else np.zeros(0)
    summary: dict = {
        "schema": SCHEMA,
        "trace": trace_path.name,
        "sidecar": Path(sidecar_path).name,
        "records": len(records),
        "decisions_checked": checked,
        "skipped": dict(skipped),
        "pool_size_mean": round(float(np.mean(pool_sizes)), 2) if pool_sizes else None,
        "pool_size_max": int(np.max(pool_sizes)) if pool_sizes else None,
        "phase_exact_ratio": round(float(np.mean(phase < 0.01)), 6) if phase.size else None,
        "phase_p95_px": round(float(np.percentile(phase, 95)), 3) if phase.size else None,
        "phase_max_px": round(float(phase.max()), 3) if phase.size else None,
        "min_clearance": None,
        "p1_clearance": None,
        "median_clearance": None,
        "frame_collisions": frame_collisions,
        "frame_collision_rate": round(frame_collisions / checked, 6) if checked else None,
        "near_misses_0_5": int(np.sum((values > 0.0) & (values <= 0.5))),
        "swept_collisions": swept_collisions,
        "swept_only_collisions": swept_only,
        "reported_safe_but_frame_collision": reported_safe_crossing,
        "reported_safe_but_frame_collision_ratio": (
            round(reported_safe_crossing / frame_collisions, 6) if frame_collisions else None
        ),
        "event_alignment": {
            "aligned_pilot_model": aligned,
            "teacher_above_model": teacher_above,
            "teacher_below_model": teacher_below,
        },
        "collisions_within_3_decisions_of_death": before_death,
        "death_decisions": death_decisions,
        "events": [
            {
                "decision": event.decision,
                "frame": event.frame,
                "frames": event.frames,
                "frame_clearance": event.frame_clearance,
                "frame_index": event.frame_index,
                "swept_clearance": event.swept_clearance,
                "pilot_model_clearance": event.pilot_model_clearance,
                "teacher_clearance": event.teacher_clearance,
            }
            for event in events
        ],
    }
    if values.size:
        summary["min_clearance"] = round(float(values.min()), 4)
        summary["p1_clearance"] = round(float(np.percentile(values, 1)), 4)
        summary["median_clearance"] = round(float(np.median(values)), 4)
    return summary
