"""``TeacherLabel``: what every decision hands downstream (11-P1 §2 step 6).

The label is deliberately plain data — a fixed 18-vector plus the chosen plan and
the verification outcome — so it can be written to JSONL and read back by P3's
DAgger loop without re-running the teacher.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from touhou_rl.v400.teacher.config import N_CAND
from touhou_rl.v400.teacher.search import Plan


@dataclass
class TeacherLabel:
    """One decision's label, in the shape 11-P1 §2 specifies.

    ``scores`` is ``geo_score[18]``: the best score each first action reached.
    Verified actions keep that score; unverified ones are discounted by
    ``weights.unverified_discount`` (11-P1 §3), and ``verified_mask`` says which
    is which, so a consumer never has to guess.
    """

    scores: np.ndarray
    bomb: int
    chosen_plan: Plan | None
    margin: float
    verified_mask: np.ndarray
    #: 11-P1 §4 rule 2 is judged from the engine's own ``EVENTS`` word.  It is
    #: ``False`` only when the engine refused the command (``ERR
    #: events-unavailable``), in which case ``frame_event_flags`` stays ``None``
    #: and no death bomb is fabricated.
    bomb_death_available: bool = False
    #: Raw ``ReplayManager::frameEventFlags`` of the sampled frame (``None`` when
    #: the engine could not answer).  ``0x2`` is ``PLAYER_HIT``.
    frame_event_flags: int | None = None
    bombs: float = 0.0
    #: 13-P3 §2.2 / D-08: every committed frame of this decision carries
    #: ``BTN_SHOOT`` (the candidate space has no shoot dimension at all).
    shoot: bool = True
    #: EXP-P1-009: this decision's block ends with exactly one frame of SHOOT
    #: released, so the next frame is the rising edge that
    #: ``GameManager.cpp:381-389`` needs to finish a stage transition.
    shoot_release: bool = False
    frame: int = 0
    stage: int = 0
    difficulty: int = 0
    events: list[str] = field(default_factory=list)
    verified: list[dict] = field(default_factory=list)
    timing_ms: dict[str, float] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=np.float64)
        self.verified_mask = np.asarray(self.verified_mask, dtype=bool)
        if self.scores.shape != (N_CAND,):
            raise ValueError(f"scores must have {N_CAND} entries, got {self.scores.shape}")
        if self.verified_mask.shape != (N_CAND,):
            raise ValueError(
                f"verified_mask must have {N_CAND} entries, got {self.verified_mask.shape}"
            )
        if self.bomb not in (0, 1):
            raise ValueError(f"bomb must be 0 or 1, got {self.bomb}")

    # -- serialisation ----------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "frame": int(self.frame),
            "stage": int(self.stage),
            "difficulty": int(self.difficulty),
            "scores": [round(float(value), 6) for value in self.scores],
            "verified_mask": [bool(value) for value in self.verified_mask],
            "bomb": int(self.bomb),
            "bomb_death_available": bool(self.bomb_death_available),
            "frame_event_flags": None
            if self.frame_event_flags is None
            else int(self.frame_event_flags),
            "bombs": float(self.bombs),
            "shoot": bool(self.shoot),
            "shoot_release": bool(self.shoot_release),
            "margin": round(float(self.margin), 4),
            "chosen_plan": None if self.chosen_plan is None else self.chosen_plan.as_dict(),
            "events": list(self.events),
            "verified": list(self.verified),
            "timing_ms": {key: round(float(value), 3) for key, value in self.timing_ms.items()},
            "extra": dict(self.extra),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict) -> "TeacherLabel":
        plan_payload = payload.get("chosen_plan")
        plan = None
        if plan_payload is not None:
            plan = Plan(
                first_action=int(plan_payload["first_action"]),
                duration=int(plan_payload["duration"]),
                slots=tuple(int(x) for x in plan_payload["slots"]),
                margin=float(plan_payload["margin"]),
                death_frame=plan_payload.get("death_frame"),
                end_xy=tuple(float(x) for x in plan_payload["end_xy"]),  # type: ignore[arg-type]
                score=float(plan_payload["score"]),
                verified=bool(plan_payload.get("verified", False)),
                align=float(plan_payload.get("align", 0.0)),
                edge=float(plan_payload.get("edge", 0.0)),
                high=float(plan_payload.get("high", 0.0)),
                items=float(plan_payload.get("items", 0.0)),
                bottom=float(plan_payload.get("bottom", 0.0)),
                escape=float(plan_payload.get("escape", 0.0)),
                boss=float(plan_payload.get("boss", 0.0)),
                top=float(plan_payload.get("top", 0.0)),
                corner=float(plan_payload.get("corner", 0.0)),
                laser=float(plan_payload.get("laser", 0.0)),
                geo_score=plan_payload.get("geo_score"),
            )
        return cls(
            scores=np.asarray(payload["scores"], dtype=np.float64),
            bomb=int(payload["bomb"]),
            chosen_plan=plan,
            margin=float(payload["margin"]),
            verified_mask=np.asarray(payload["verified_mask"], dtype=bool),
            bomb_death_available=bool(payload.get("bomb_death_available", False)),
            frame_event_flags=(
                None
                if payload.get("frame_event_flags") is None
                else int(payload["frame_event_flags"])
            ),
            bombs=float(payload.get("bombs", 0.0)),
            shoot=bool(payload.get("shoot", True)),
            shoot_release=bool(payload.get("shoot_release", False)),
            frame=int(payload.get("frame", 0)),
            stage=int(payload.get("stage", 0)),
            difficulty=int(payload.get("difficulty", 0)),
            events=list(payload.get("events", [])),
            verified=list(payload.get("verified", [])),
            timing_ms=dict(payload.get("timing_ms", {})),
            extra=dict(payload.get("extra", {})),
        )

    # -- helpers ----------------------------------------------------------
    def with_verification(self, result, verified_plans, weights) -> "TeacherLabel":
        """Apply 11-P1 §3's verified/unverified split to ``scores``.

        Verified first actions keep their engine-verified score; every other
        action is discounted by ``weights.unverified_discount``.  The mask is
        updated in the same pass so the two can never disagree.
        """
        scores = np.array(self.scores, dtype=np.float64, copy=True)
        mask = np.zeros(N_CAND, dtype=bool)
        for verified in verified_plans:
            cand = int(verified.plan.first_action)
            scores[cand] = float(verified.plan.score)
            mask[cand] = True
        for cand in range(N_CAND):
            if not mask[cand] and np.isfinite(scores[cand]):
                scores[cand] -= float(weights.unverified_discount)
        self.scores = scores
        self.verified_mask = mask
        self.verified = [plan.as_dict() for plan in verified_plans]
        return self
