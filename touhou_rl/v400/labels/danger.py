"""Future-danger labels: "would the ship die if it did not move?" (P2 §3.3).

The geometry is the model EXP-P1-004 validated against the engine:
``geometry.MODEL_ENGINE_PHASE`` — bullets advanced exactly one tick by their
``velocity``, the hurtbox the engine holds at the test moment, ``collisionSize/
2`` half widths, ``hitboxDimensions/3`` enemy bodies, flag gating, inclusive
boundaries, float32.  Nothing here re-derives an AABB; the whole computation is
:func:`touhou_rl.v400.teacher.geometry.frame_verdict` over the recorded future
hazard frames, with the self ship frozen at its current position.

What the label means, precisely:

* **Counterfactual, not prediction of the recorded run.**  The hazard snapshots
  are the ones the real run produced; the ship is held still.  Hazards that aim
  at the ship are not modelled (this is an approximation, recorded as such).
* **Fail-closed.**  The label only has a value while ``playerState == ALIVE``;
  on DYING / SPAWNING / INVULNERABLE frames it is ``None`` with a reason,
  because "would die" is not a meaningful question there.  A run tail with
  fewer than H future frames gives ``per_horizon[h] = None`` for uncovered h
  instead of a guessed ``False``.
* **Same frame alignment as EXP-P1-004.**  For offset ``j`` (1..H) the hazard
  frame is ``k+j`` and the hurtbox is the frozen one; the model's one-tick
  bullet advance stays inside the model, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Sequence

from touhou_rl.v400.labels.spec import HORIZON_FRAMES, HORIZON_HEADLINE
from touhou_rl.v400.labels.state import (
    PLAYER_STATE_ALIVE,
    PLAYER_STATE_NAMES,
)
from touhou_rl.v400.teacher import geometry

DEFAULT_HORIZONS = HORIZON_FRAMES


@dataclass(frozen=True)
class FutureDanger:
    """One frame's future-danger label."""

    horizon: int
    would_die: bool | None
    first_hit_offset: int | None
    min_gap: float | None
    per_horizon: dict[int, bool | None]
    frames_considered: int
    frames_with_hazard: int
    applicable: bool
    reason: str
    model: str = geometry.MODEL_ENGINE_PHASE.name
    still: bool = True
    hit_offsets: tuple[int, ...] = ()
    missing: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "would_die": self.would_die,
            "first_hit_offset": self.first_hit_offset,
            "min_gap": self.min_gap,
            "per_horizon": {str(key): value for key, value in self.per_horizon.items()},
            "frames_considered": self.frames_considered,
            "frames_with_hazard": self.frames_with_hazard,
            "applicable": self.applicable,
            "reason": self.reason,
            "model": self.model,
            "still": self.still,
            "hit_offsets": list(self.hit_offsets),
            "missing": list(self.missing),
        }


def _frozen_ship(player: Any) -> SimpleNamespace | None:
    """A stand-in ``PLAYER`` holding still: same hurtbox, no other state needed."""

    hurt_min = getattr(player, "hurtbox_min", None)
    hurt_max = getattr(player, "hurtbox_max", None)
    if hurt_min is None or hurt_max is None:
        return None
    return SimpleNamespace(
        hurtbox_min=(float(hurt_min[0]), float(hurt_min[1])),
        hurtbox_max=(float(hurt_max[0]), float(hurt_max[1])),
    )


def evaluate_future_danger(
    player: Any,
    future_hazards: Sequence[Any],
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    horizon: int = HORIZON_HEADLINE,
    model: geometry.Model = geometry.MODEL_ENGINE_PHASE,
) -> FutureDanger:
    """Evaluate the still-ship counterfactual over ``future_hazards``.

    ``future_hazards[j-1]`` is the hazard snapshot of frame ``k+j`` (j = 1..H),
    i.e. the caller passes the *recorded successors*, not copies of frame k.
    """

    ordered = tuple(sorted({int(h) for h in horizons} | {int(horizon)}))
    available = len(future_hazards)
    player_state = getattr(player, "player_state", None)
    per_horizon: dict[int, bool | None] = {h: None for h in ordered}

    if player_state is None:
        return FutureDanger(
            horizon=int(horizon), would_die=None, first_hit_offset=None,
            min_gap=None, per_horizon=per_horizon, frames_considered=0,
            frames_with_hazard=0, applicable=False,
            reason="player_state_missing", model=model.name,
            missing=("player_state",),
        )
    if int(player_state) != PLAYER_STATE_ALIVE:
        name = PLAYER_STATE_NAMES.get(int(player_state), str(player_state))
        return FutureDanger(
            horizon=int(horizon), would_die=None, first_hit_offset=None,
            min_gap=None, per_horizon=per_horizon, frames_considered=0,
            frames_with_hazard=0, applicable=False,
            reason=f"player_state_{name}", model=model.name,
        )

    ship = _frozen_ship(player)
    if ship is None:
        return FutureDanger(
            horizon=int(horizon), would_die=None, first_hit_offset=None,
            min_gap=None, per_horizon=per_horizon, frames_considered=0,
            frames_with_hazard=0, applicable=False,
            reason="hurtbox_missing", model=model.name,
            missing=("hurtbox_min", "hurtbox_max"),
        )

    hit_offsets: list[int] = []
    gaps: list[float] = []
    frames_with_hazard = 0
    considered = min(available, max(ordered))
    for offset in range(1, considered + 1):
        hazards = future_hazards[offset - 1]
        if len(getattr(hazards, "bullets", ())) or len(getattr(hazards, "enemies", ())) \
                or len(getattr(hazards, "lasers", ())):
            frames_with_hazard += 1
        verdict = geometry.frame_verdict(hazards, ship, model)
        if verdict.hit:
            hit_offsets.append(offset)
            gaps.append(0.0)
        elif verdict.nearest is not None:
            gaps.append(float(verdict.nearest[2]))

    for h in ordered:
        if h <= considered:
            per_horizon[h] = any(offset <= h for offset in hit_offsets)
        # else: None, the run tail does not cover this horizon

    would_die = per_horizon.get(int(horizon))
    first_hit = min(hit_offsets) if hit_offsets else None
    min_gap = min(gaps) if gaps else None
    reason = ""
    if would_die is None:
        reason = "insufficient_future_frames"
    elif hit_offsets:
        reason = "hit_within_horizon"

    return FutureDanger(
        horizon=int(horizon),
        would_die=would_die,
        first_hit_offset=first_hit,
        min_gap=min_gap,
        per_horizon=per_horizon,
        frames_considered=considered,
        frames_with_hazard=frames_with_hazard,
        applicable=True,
        reason=reason,
        model=model.name,
        hit_offsets=tuple(hit_offsets),
    )
