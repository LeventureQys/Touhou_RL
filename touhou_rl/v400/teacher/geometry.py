"""Exact per-frame hazard geometry for the P1 teacher (11-P1 §2.1).

This module answers exactly one question, in numpy, with no engine access:

    does the self ship's hurtbox AABB overlap any bullet / laser / enemy-body
    AABB in this frame's :class:`HazardFrame`?

Everything here is derived from engine source and from the raw numbers the
engine exports (``HAZARDS`` + ``PLAYER``); nothing is re-scaled a second time.
The three rules, and where they come from:

* **Bullet** —``state == BULLET_STATE_FIRED`` and ``collisionDisabled == 0``.
  The exported ``collisionSize`` is the **full** width
  (``Bullet::sprites.collisionSize.x``); ``Player::CheckBulletCollision``
  (Player.cpp:309-336) builds ``position ± size / 2``, so the lethal half
  extent is ``collisionSize / 2``.  Treating it as a half width is the
  documented two-times error of the old teacher (11-P1 §1).
* **Enemy body** —``flags1`` bit 2 (``ENEMY_FLAG_COLLISION``) and the
  **``worldPosition``** centre, not ``position``.  ``Enemy::CheckPlayerCollision``
  divides the raw ``hitboxDimensions`` by 1.5 *and then* hands it to
  ``CheckLethalCollision``, which halves it again, so the lethal half extent is
  ``hitboxDimensions / 3`` (EnemyManager.cpp:851-853, Player.cpp:339-350).
* **Laser** —``Player::CalcLaserHitbox`` (Player.cpp:416-440).  The self ship is
  rotated *into* the laser frame (``origin + R(-angle) * (p - origin)``) and then
  tested against an axis-aligned rectangle whose centre is
  ``(origin.x + start + (end-start)/2, origin.y)``, whose half length is
  ``(end - start) / 2`` (times 0.7 when ``startOffset > 0``) and whose half
  thickness is ``width / 4`` (the engine sets ``size.y = width / 2`` and then
  halves ``size``).

Boundary semantics are **inclusive**: the engine returns "no hit" only when
``hurtboxMin.x > boxMax.x`` (or the analogous strict comparison on the other
three sides), so a box that merely touches the hurtbox does hit.  All
comparisons are therefore ``<=`` / ``>=`` and never ``<`` / ``>``.

Precision: the engine is compiled for x86 and does this arithmetic in
``float``.  :data:`MODEL_SPEC` reproduces that with ``numpy.float32`` rather
than widening to ``float64``, because the numbers involved (a 1.65 px hurtbox
against 4.0 px bullets) are close enough to the boundary that double rounding
can flip a touched box.  ``float64`` is kept selectable so the difference can
be measured instead of assumed.

Positions arrive as ``%08x`` float bits, i.e. exactly the ``float32`` the
engine holds, so the only rounding is the one the engine itself did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

#: ``BulletState::BULLET_STATE_FIRED`` —the only bullet state that can kill.
BULLET_STATE_FIRED = 1

#: ``ENEMY_FLAG_COLLISION`` —``flags1`` bit 2 (``EnemyFlag1Bits::collision``).
ENEMY_FLAG_COLLISION = 1 << 2

#: ``EnemyFlag1Bits`` bits that *suppress* the enemy-body lethal test even when
#: ``collision`` is set.  EnemyManagerUpdate.cpp:287-291 requires all three to
#: be clear before it calls ``CheckPlayerCollision``, so a model that only
#: looks at bit 2 over-reports lethal bodies.
ENEMY_FLAG_NO_SPRITE = 1 << 4
ENEMY_FLAG_HIDE_PRIMARY_ANM = 1 << 5
ENEMY_FLAG_YOUKAI_ALIGNED = 1 << 11

#: ``EnemyFlag1Bits::noDamageDuringStop`` —disables the body hitbox while a
#: bomb is in use.  The bomb state is not part of a frame record here, so the
#: model assumes no bomb (true for every run in EXP-P1-004: BOMB is never sent).
ENEMY_FLAG_NO_DAMAGE_DURING_STOP = 1 << 27

#: Every ``flags1`` bit that must be clear for the body hitbox to be lethal.
ENEMY_FLAG_SUPPRESSING = (
    ENEMY_FLAG_NO_SPRITE | ENEMY_FLAG_HIDE_PRIMARY_ANM | ENEMY_FLAG_YOUKAI_ALIGNED
)

#: ``LaserState`` (BulletManager.hpp:306-311).  ``LASER_STATE_ACTIVE`` is the
#: only state a pre-EXP-P1-021 row can recognise; the other two carry hitbox
#: windows the engine now exports explicitly (see :func:`laser_hits`).
LASER_STATE_STARTING = 0
LASER_STATE_ACTIVE = 1
LASER_STATE_DESPAWNING = 2

#: Kind codes, stable so a report can group mismatches by hazard family.
KIND_BULLET = 0
KIND_LASER = 1
KIND_ENEMY = 2
KIND_NAMES = {KIND_BULLET: "bullet", KIND_LASER: "laser", KIND_ENEMY: "enemyBody"}


@dataclass(frozen=True)
class Model:
    """One geometric model, so a wrong one can be measured against a right one.

    ``MODEL_SPEC`` is the 11-P1 §2.1 model.  ``MODEL_LEGACY_HALF_WIDTH`` is the
    bug that model replaces —``collisionSize`` read as a half width and
    ``hitboxDimensions / 1.5`` read as a half extent, with strict (exclusive)
    boundary comparisons.  Keeping it addressable is what makes "before and
    after the fix" a measurement on the same recorded frames instead of a
    recollection.
    """

    name: str
    bullet_half_scale: float
    enemy_half_divisor: float
    inclusive: bool
    dtype: str = "float32"
    #: Gate the enemy body on every ``flags1`` bit the engine checks, or only on
    #: ``collision`` (bit 2) as 11-P1 §2.1 literally reads.
    enemy_flag_gate: bool = True
    #: How many ticks of ``velocity`` to add to a bullet before testing it.
    #:
    #: 0.0 is "test the snapshot you just read".  The engine does **not** work
    #: that way: in ``BULLET_STATE_FIRED`` it runs ``position += velocity``
    #: (BulletManager.cpp:862-863) and only then reaches the collision block
    #: (BulletManager.cpp:899-950), so its test point is one tick **ahead** of
    #: the position a snapshot taken at the end of that same frame can show.
    #: Worse, the bullet that hits is set to ``BULLET_STATE_DESPAWNING`` inside
    #: that same block, so it is gone from the *next* snapshot too: the killing
    #: bullet is invisible in both frames it spans.  1.0 reconstructs it from
    #: the velocity the row already carries.
    bullet_advance: float = 0.0

    def np_dtype(self):
        return np.dtype(self.dtype)


MODEL_SPEC = Model(
    name="spec",
    bullet_half_scale=0.5,
    enemy_half_divisor=3.0,
    inclusive=True,
    dtype="float32",
)

#: ``spec`` plus the one-tick advance the engine performs before testing.
#: Pair it with the **next** frame's hurtbox (the hurtbox the engine holds when
#: it tests); :func:`frame_verdict` takes the hurtbox from the snapshot it is
#: given, so the caller passes ``player(k+1)``.
MODEL_ENGINE_PHASE = Model(
    name="engine-phase",
    bullet_half_scale=0.5,
    enemy_half_divisor=3.0,
    inclusive=True,
    dtype="float32",
    bullet_advance=1.0,
)

MODEL_COLLISION_BIT_ONLY = Model(
    name="collision-bit-only",
    bullet_half_scale=0.5,
    enemy_half_divisor=3.0,
    inclusive=True,
    dtype="float32",
    enemy_flag_gate=False,
)

MODEL_LEGACY_HALF_WIDTH = Model(
    name="legacy-half-width",
    bullet_half_scale=1.0,
    enemy_half_divisor=1.5,
    inclusive=False,
    dtype="float32",
)

MODELS = {
    model.name: model
    for model in (
        MODEL_SPEC,
        MODEL_ENGINE_PHASE,
        MODEL_COLLISION_BIT_ONLY,
        MODEL_LEGACY_HALF_WIDTH,
    )
}

#: Which models are evaluated against the frame's own hurtbox and which need
#: the next frame's.  ``engine-phase`` tests at the point the engine tests at,
#: so it has to be given the hurtbox the engine holds at that moment.
HURTBOX_PHASE = {
    "spec": "current",
    "engine-phase": "next",
    "collision-bit-only": "current",
    "legacy-half-width": "current",
}


# ---------------------------------------------------------------------------
# self-ship movement speeds
# ---------------------------------------------------------------------------
#: ``PLAYER`` reply keys -> :class:`SpeedTable` fields.  TH08 reads these four
#: numbers out of the loaded ``.sht`` file (Player.cpp:783-806 uses
#: ``primaryShtFile`` unfocused and ``secondaryShtFile`` focused), so they are
#: per-shot data, never constants.
PLAYER_SPEED_FIELDS = (
    ("speed_normal_axial", "normal_axial"),
    ("speed_normal_diagonal", "normal_diagonal"),
    ("speed_focused_axial", "focused_axial"),
    ("speed_focused_diagonal", "focused_diagonal"),
)


@dataclass(frozen=True)
class SpeedTable:
    """The four self-ship speeds ``PLAYER`` reports, in px/frame.

    This is the **single** definition; :mod:`touhou_rl.v400.teacher.motion`
    re-exports it for the movement model and
    :mod:`touhou_rl.v400.teacher.hazard_tape` imports it from here, so there is
    one class and not two classes that happen to have the same fields.

    The unfocused pair is ``normal_axial``/``normal_diagonal`` because that is
    what the engine calls it (``normalAxisSpeed``); :attr:`axial` and
    :attr:`diagonal` are the same two numbers under the shorter names the spec
    uses.  A diagonal step moves by ``diagonal`` on **both** axes
    (:func:`touhou_rl.v400.teacher.motion.step_delta`).
    """

    normal_axial: float
    normal_diagonal: float
    focused_axial: float
    focused_diagonal: float

    @classmethod
    def from_player(cls, player) -> "SpeedTable":
        """Build the table from a ``PLAYER`` reply (or any object shaped like one).

        Missing fields are an error, never a default: a zero speed would make
        the search believe the ship cannot move.  Positivity is left to
        :meth:`check` because a caller may want to report *which* speed the
        engine failed to load before failing.
        """

        missing = [
            wire_name
            for wire_name, _ in PLAYER_SPEED_FIELDS
            if getattr(player, wire_name, None) is None
        ]
        if missing:
            raise ValueError(f"PLAYER snapshot is missing speed fields: {missing}")
        return cls(
            *(
                float(getattr(player, wire_name))
                for wire_name, _ in PLAYER_SPEED_FIELDS
            )
        )

    @property
    def axial(self) -> float:
        """Unfocused single-axis speed (alias of :attr:`normal_axial`)."""

        return self.normal_axial

    @property
    def diagonal(self) -> float:
        """Unfocused diagonal component (alias of :attr:`normal_diagonal`)."""

        return self.normal_diagonal

    def check(self) -> None:
        for field_name in ("normal_axial", "normal_diagonal", "focused_axial", "focused_diagonal"):
            value = getattr(self, field_name)
            if value is None or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"speed {field_name} is not a positive finite number: {value!r}")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.normal_axial,
            self.normal_diagonal,
            self.focused_axial,
            self.focused_diagonal,
        )


# ---------------------------------------------------------------------------
# hazard AABBs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Boxes:
    """A batch of hazard AABBs plus the ``(kind, slot)`` that produced each one."""

    mins: np.ndarray  # (N, 2) [x, y]
    maxs: np.ndarray  # (N, 2)
    kind: np.ndarray  # (N,) int8, KIND_*
    slot: np.ndarray  # (N,) int32

    def __len__(self) -> int:
        return int(self.mins.shape[0])

    @classmethod
    def empty(cls, dtype: str = "float32") -> "Boxes":
        dt = np.dtype(dtype)
        return cls(
            mins=np.zeros((0, 2), dtype=dt),
            maxs=np.zeros((0, 2), dtype=dt),
            kind=np.zeros((0,), dtype=np.int8),
            slot=np.zeros((0,), dtype=np.int32),
        )


def bullet_boxes(bullets: Iterable, model: Model = MODEL_SPEC) -> Boxes:
    """Lethal bullet AABBs (``state == FIRED`` and ``collisionDisabled == 0``).

    A bullet whose ``collision_size`` is missing (pre-P1 engine) is dropped
    rather than guessed: a missing width is not a zero width.
    """

    dt = model.np_dtype()
    keep_min = []
    keep_max = []
    keep_slot = []
    advance = dt.type(model.bullet_advance)
    for bullet in bullets:
        if getattr(bullet, "state", None) != BULLET_STATE_FIRED:
            continue
        if getattr(bullet, "collision_disabled", None):
            continue
        size = getattr(bullet, "collision_size", None)
        if size is None:
            continue
        half = dt.type(size * model.bullet_half_scale)
        x, y = bullet.position[0], bullet.position[1]
        if advance != 0.0:
            # Same float32 add the engine does, in the same order.
            velocity = getattr(bullet, "velocity", None)
            if velocity is None:
                raise ValueError(
                    "model needs bullet velocity to advance a tick, but the row "
                    "has none; record BULLETS fields 6-7 (velXy)"
                )
            x = dt.type(x) + dt.type(velocity[0]) * advance
            y = dt.type(y) + dt.type(velocity[1]) * advance
        # TH08 assigns collisionSize.x == collisionSize.y for every bullet type,
        # so the single exported width drives both axes (EXP-P1-001 §3).
        keep_min.append((dt.type(x) - half, dt.type(y) - half))
        keep_max.append((dt.type(x) + half, dt.type(y) + half))
        keep_slot.append(int(bullet.slot))
    if not keep_min:
        return Boxes.empty(model.dtype)
    return Boxes(
        mins=np.asarray(keep_min, dtype=dt),
        maxs=np.asarray(keep_max, dtype=dt),
        kind=np.full((len(keep_min),), KIND_BULLET, dtype=np.int8),
        slot=np.asarray(keep_slot, dtype=np.int32),
    )


def enemy_boxes(enemies: Iterable, model: Model = MODEL_SPEC) -> Boxes:
    """Lethal enemy-body AABBs (``flags1`` bit 2, centred on ``worldPosition``).

    ``worldPosition`` is mandatory: a parented enemy's ``position`` is the
    stage-local value and is not where its body actually is (EXP-P1-001 §3).
    An enemy without an exported ``worldPosition`` or ``hitbox`` is dropped.
    """

    dt = model.np_dtype()
    keep_min = []
    keep_max = []
    keep_slot = []
    for enemy in enemies:
        flags = int(getattr(enemy, "flags1", 0))
        if not flags & ENEMY_FLAG_COLLISION:
            continue
        if model.enemy_flag_gate and flags & ENEMY_FLAG_SUPPRESSING:
            continue
        world = getattr(enemy, "world_position", None)
        width = getattr(enemy, "hitbox_width", None)
        height = getattr(enemy, "hitbox_height", None)
        if world is None or width is None or height is None:
            continue
        half_x = dt.type(width / model.enemy_half_divisor)
        half_y = dt.type(height / model.enemy_half_divisor)
        cx, cy = world[0], world[1]
        keep_min.append((dt.type(cx) - half_x, dt.type(cy) - half_y))
        keep_max.append((dt.type(cx) + half_x, dt.type(cy) + half_y))
        keep_slot.append(int(enemy.slot))
    if not keep_min:
        return Boxes.empty(model.dtype)
    return Boxes(
        mins=np.asarray(keep_min, dtype=dt),
        maxs=np.asarray(keep_max, dtype=dt),
        kind=np.full((len(keep_min),), KIND_ENEMY, dtype=np.int8),
        slot=np.asarray(keep_slot, dtype=np.int32),
    )


@dataclass(frozen=True)
class LaserHits:
    """Result of the laser family, which is not an axis-aligned world query."""

    hit: bool
    #: ``(kind, slot, gap)`` of the closest laser, for the mismatch table.
    nearest: tuple[int, int, float] | None = None
    #: True when at least one laser was considered at all.
    considered: int = 0
    #: Lasers that were in use but whose state carries no hitbox this revision.
    skipped_inactive: int = 0


def _rotate_into_laser_frame(
    point: tuple[float, float],
    origin: tuple[float, float],
    angle: float,
    dtype,
) -> tuple[float, float]:
    """``origin + R(-angle) * (point - origin)``, exactly as the engine does it.

    ``Player::CalcLaserHitbox`` subtracts the laser origin, rotates the
    difference by ``-angle`` and adds the origin back; the result is compared
    against a rectangle that lives in the same unrotated frame.
    """

    dx = dtype.type(point[0]) - dtype.type(origin[0])
    dy = dtype.type(point[1]) - dtype.type(origin[1])
    cos = dtype.type(math.cos(-angle))
    sin = dtype.type(math.sin(-angle))
    # ZUN's Rotate() is the standard 2D rotation; the sign convention only has
    # to match the engine's, and it does because it is the same call.
    rx = dx * cos - dy * sin
    ry = dx * sin + dy * cos
    return (dtype.type(origin[0]) + rx, dtype.type(origin[1]) + ry)


def _laser_hitbox(laser, dt) -> tuple[float, float, float, float] | None:
    """``(origin_x, origin_y, center_x, half_len)`` of one laser, or ``None``.

    ``None`` means "this laser is not a hazard on the frame the row describes":
    either the engine's ``damageable`` flag is 0, or (a pre-EXP-P1-021 row with
    no flag) its state is not ``LASER_STATE_ACTIVE``.

    The expression is the engine's own, byte for byte the arithmetic of
    ``Player::CalcLaserHitbox`` / ``BulletManager::OnUpdate``:

    * ``size.y = width / 2`` then the player test halves ``size`` again, hence
      the ``width / 4`` half thickness (BulletManager.cpp:1093, Player.cpp:428);
    * ``size.x = (end - start)`` —times ``0.7`` once ``startOffset > 0`` —and
      the centre is ``origin.x + start + (end - start) / 2``
      (BulletManager.cpp:1094-1100, Player.cpp:424-432).
    """

    if getattr(laser, "in_use", None) == 0:
        return None
    damageable = getattr(laser, "damageable", None)
    if damageable is None:
        if int(getattr(laser, "state", -1)) != LASER_STATE_ACTIVE:
            return None
    elif not int(damageable):
        return None
    return laser_rectangle(laser, dt)


def laser_rectangle(laser, dt) -> tuple[float, float, float, float]:
    """``(origin_x, origin_y, center_x, half_len)`` of a laser's hit rectangle."""

    start = float(laser.start_offset)
    end = float(laser.end_offset)
    length = dt.type(end - start)
    if start <= 0.0:
        half_len = length / dt.type(2.0)
    else:
        # The engine scales the *length* by 0.7 once the beam has pulled its
        # start off the origin; the half length keeps the same factor.
        half_len = dt.type(length * dt.type(0.7)) / dt.type(2.0)
    origin = (float(laser.position[0]), float(laser.position[1]))
    cx = dt.type(origin[0]) + dt.type(start) + length / dt.type(2.0)
    return (origin[0], origin[1], float(cx), float(half_len))


def laser_half_thickness(laser, dt) -> float:
    """``width / 4``: the laser's lethal half thickness (Player.cpp:428)."""

    return float(dt.type(float(laser.width)) / dt.type(4.0))


def laser_hits(
    lasers: Iterable,
    hurtbox_min: Sequence[float],
    hurtbox_max: Sequence[float],
    model: Model = MODEL_SPEC,
) -> LaserHits:
    """Exact copy of ``Player::CalcLaserHitbox`` for every damageable laser.

    The self ship is rotated into the laser frame and tested as an
    axis-aligned square of half size ``(hurtboxMax - hurtboxMin) / 2`` against
    the laser rectangle.  That is not an approximation of the engine: the
    engine does not rotate the hurtbox either, it translates it.

    Which lasers count (EXP-P1-021): the engine's ``damageable`` flag, i.e. the
    frame is inside the state's judgement window —
    * ``LASER_STATE_STARTING``: ``timer >= hitboxStartTime``;
    * ``LASER_STATE_ACTIVE``: always;
    * ``LASER_STATE_DESPAWNING``: ``timer < hitboxEndDelay``

    (BulletManager.cpp:1106-1142).  A row recorded by a pre-EXP-P1-021 engine
    has no flag, so the old state-only rule is used and those lasers are
    counted in :attr:`LaserHits.skipped_inactive` when they are not
    ``LASER_STATE_ACTIVE`` —the fallback is deliberate, not a guess: it is the
    rule this module shipped with, and it under-reports rather than inventing
    a hitbox.
    """

    dt = model.np_dtype()
    half_x = (dt.type(hurtbox_max[0]) - dt.type(hurtbox_min[0])) / dt.type(2.0)
    half_y = (dt.type(hurtbox_max[1]) - dt.type(hurtbox_min[1])) / dt.type(2.0)
    center = (
        (dt.type(hurtbox_min[0]) + dt.type(hurtbox_max[0])) / dt.type(2.0),
        (dt.type(hurtbox_min[1]) + dt.type(hurtbox_max[1])) / dt.type(2.0),
    )

    any_hit = False
    nearest: tuple[int, int, float] | None = None
    considered = 0
    skipped = 0
    for laser in lasers:
        if getattr(laser, "in_use", None) == 0:
            continue
        box = _laser_hitbox(laser, dt)
        if box is None:
            skipped += 1
            continue
        considered += 1
        origin = (box[0], box[1])
        cx, half_len = box[2], box[3]
        cy = dt.type(box[1])
        half_thick = dt.type(laser_half_thickness(laser, dt))

        px, py = _rotate_into_laser_frame(center, origin, float(laser.angle), dt)
        dx = _axis_gap(px, half_x, cx, half_len, dt)
        dy = _axis_gap(py, half_y, cy, half_thick, dt)
        gap = float(dt.type(math.hypot(dx, dy)))
        if nearest is None or gap < nearest[2]:
            nearest = (KIND_LASER, int(laser.slot), gap)
        if _within(dx, dy, model.inclusive):
            any_hit = True

    return LaserHits(
        hit=any_hit, nearest=nearest, considered=considered, skipped_inactive=skipped
    )


def laser_gap_rows(
    lasers: Iterable,
    centers: np.ndarray,
    half: Sequence[float],
    model: Model = MODEL_SPEC,
) -> np.ndarray:
    """Min pixel gap from each terminal centre to any **damageable** laser.

    ``laser_hits`` answers "does this ship overlap" for one point; the beam
    search needs the same question for a whole batch of candidate terminal
    positions, and it needs a *distance* rather than a boolean so the score can
    grade it.  Both functions share :func:`_laser_hitbox` —the ``damageable``
    gate and the rectangle arithmetic —so the search's laser term and the
    frame verdict cannot disagree about which lasers are hazards.

    ``0.0`` means the hurtbox (an AABB of half size ``half`` around the centre)
    touches the laser rectangle; ``inf`` means no damageable laser exists on
    that frame.

    Precision: this is a cost term, not a boundary decision, so it evaluates in
    ``float64``.  :func:`laser_hits` keeps the engine's ``float32`` because it
    must reproduce a hit/no-hit boundary; the two differ only in the last bits
    of a distance, never in the damageable gate.
    """

    del model  # kept for signature symmetry with ``laser_hits``
    batch = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    out = np.full((batch.shape[0],), np.inf, dtype=np.float64)
    if batch.shape[0] == 0:
        return out
    half_x = float(half[0])
    half_y = float(half[1])
    for laser in lasers:
        if getattr(laser, "in_use", None) == 0:
            continue
        _gate = _laser_hitbox(laser, np.dtype(np.float64))
        if _gate is None:
            continue
        origin_x, origin_y, cx, half_len = _gate
        half_thick = float(laser.width) / 4.0
        angle = float(laser.angle)
        cos = math.cos(-angle)
        sin = math.sin(-angle)
        dx = batch[:, 0] - origin_x
        dy = batch[:, 1] - origin_y
        px = origin_x + (dx * cos - dy * sin)
        py = origin_y + (dx * sin + dy * cos)
        gap_x = np.maximum(np.abs(px - cx) - (half_x + half_len), 0.0)
        gap_y = np.maximum(np.abs(py - origin_y) - (half_y + half_thick), 0.0)
        np.minimum(out, np.hypot(gap_x, gap_y), out=out)
    return out


def _axis_gap(center_a, half_a, center_b, half_b, dt) -> float:
    """Separation of two intervals on one axis, 0 when they overlap or touch."""

    delta = abs(dt.type(center_a) - dt.type(center_b))
    slack = dt.type(half_a) + dt.type(half_b)
    gap = dt.type(delta - slack)
    return 0.0 if float(gap) <= 0.0 else float(gap)


def _within(dx: float, dy: float, inclusive: bool) -> bool:
    if inclusive:
        return dx <= 0.0 and dy <= 0.0
    return dx < 0.0 and dy < 0.0


# ---------------------------------------------------------------------------
# the frame verdict
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameVerdict:
    """One frame's geometric decision, with enough detail to audit a mismatch."""

    hit: bool
    #: Closest hazard ``(kind, slot, gap)``; gap is 0 when it overlaps.
    nearest: tuple[int, int, float] | None
    bullets: int
    enemies: int
    lasers: int
    lasers_considered: int
    lasers_skipped: int
    #: Which families actually produced a hit this frame.
    hit_kinds: tuple[int, ...] = ()

    def as_dict(self) -> dict:
        return {
            "hit": bool(self.hit),
            "nearest": (
                None
                if self.nearest is None
                else {
                    "kind": KIND_NAMES.get(self.nearest[0], str(self.nearest[0])),
                    "slot": int(self.nearest[1]),
                    "gap": float(self.nearest[2]),
                }
            ),
            "bullets": int(self.bullets),
            "enemies": int(self.enemies),
            "lasers": int(self.lasers),
            "lasers_considered": int(self.lasers_considered),
            "lasers_skipped": int(self.lasers_skipped),
            "hit_kinds": [KIND_NAMES.get(k, str(k)) for k in self.hit_kinds],
        }


def aabb_overlap(
    hurtbox_min: Sequence[float],
    hurtbox_max: Sequence[float],
    boxes: Boxes,
    model: Model = MODEL_SPEC,
) -> tuple[bool, tuple[int, int, float] | None, int | None]:
    """Inclusive-AABB test of one hurtbox against a batch of hazard boxes.

    Returns ``(hit, nearest, first_hitting_kind)``.  The test is written in the
    engine's own reject form —``hurtMin > boxMax`` or ``hurtMax < boxMin``
    means *no* hit —so "touching counts" falls out of the comparison rather
    than out of an epsilon.
    """

    if len(boxes) == 0:
        return (False, None, None)

    dt = model.np_dtype()
    hmin = np.asarray(hurtbox_min, dtype=dt)
    hmax = np.asarray(hurtbox_max, dtype=dt)

    with np.errstate(invalid="ignore"):
        separated = (
            (hmin[0] > boxes.maxs[:, 0])
            | (hmin[1] > boxes.maxs[:, 1])
            | (hmax[0] < boxes.mins[:, 0])
            | (hmax[1] < boxes.mins[:, 1])
            if model.inclusive
            else (
                (hmin[0] >= boxes.maxs[:, 0])
                | (hmin[1] >= boxes.maxs[:, 1])
                | (hmax[0] <= boxes.mins[:, 0])
                | (hmax[1] <= boxes.mins[:, 1])
            )
        )

    # Separation per box, for the "nearest lethal object" column.
    dx = np.maximum(boxes.mins[:, 0] - hmax[0], hmin[0] - boxes.maxs[:, 0])
    dy = np.maximum(boxes.mins[:, 1] - hmax[1], hmin[1] - boxes.maxs[:, 1])
    dx = np.maximum(dx, 0.0).astype(dt)
    dy = np.maximum(dy, 0.0).astype(dt)
    gaps = np.hypot(dx, dy).astype(dt)

    overlapping = ~separated
    if bool(overlapping.any()):
        idx = int(np.flatnonzero(overlapping)[0])
        return (True, (int(boxes.kind[idx]), int(boxes.slot[idx]), 0.0), int(boxes.kind[idx]))

    idx = int(np.argmin(gaps))
    return (False, (int(boxes.kind[idx]), int(boxes.slot[idx]), float(gaps[idx])), None)


def frame_verdict(
    hazards,
    player,
    model: Model = MODEL_SPEC,
) -> FrameVerdict:
    """11-P1 §2.1 verdict for one ``HAZARDS`` + ``PLAYER`` pair.

    ``hazards`` is a :class:`touhou_rl.v400.gym.protocol.HazardFrame` and
    ``player`` a :class:`~touhou_rl.v400.gym.protocol.PlayerSnapshot`; both are
    duck-typed so a test can pass plain stand-ins without starting an engine.
    """

    if player.hurtbox_min is None or player.hurtbox_max is None:
        raise ValueError("player snapshot has no hurtboxBounds; cannot judge geometry")

    bullets = bullet_boxes(hazards.bullets, model)
    enemies = enemy_boxes(hazards.enemies, model)

    parts = [boxes for boxes in (bullets, enemies) if len(boxes)]
    if parts:
        mins = np.concatenate([boxes.mins for boxes in parts])
        maxs = np.concatenate([boxes.maxs for boxes in parts])
        kinds = np.concatenate([boxes.kind for boxes in parts])
        slots = np.concatenate([boxes.slot for boxes in parts])
        combined = Boxes(mins=mins, maxs=maxs, kind=kinds, slot=slots)
        hit, nearest, _ = aabb_overlap(player.hurtbox_min, player.hurtbox_max, combined, model)
    else:
        combined = Boxes.empty(model.dtype)
        hit, nearest, _ = (False, None, None)

    lasers = laser_hits(
        hazards.lasers, player.hurtbox_min, player.hurtbox_max, model
    )
    hit_kinds: list[int] = []
    if hit and nearest is not None:
        hit_kinds.append(nearest[0])

    if lasers.hit:
        hit = True
        hit_kinds.append(KIND_LASER)
        if nearest is None or nearest[2] > 0.0:
            nearest = lasers.nearest

    if not hit and lasers.nearest is not None:
        if nearest is None or lasers.nearest[2] < nearest[2]:
            nearest = lasers.nearest

    return FrameVerdict(
        hit=bool(hit),
        nearest=nearest,
        bullets=len(bullets),
        enemies=len(enemies),
        lasers=len(hazards.lasers),
        lasers_considered=lasers.considered,
        lasers_skipped=lasers.skipped_inactive,
        hit_kinds=tuple(dict.fromkeys(hit_kinds)),
    )


def hazard_box_count(hazards, model: Model = MODEL_SPEC) -> int:
    """How many AABBs the model actually tested, for a sample-volume check."""

    return len(bullet_boxes(hazards.bullets, model)) + len(enemy_boxes(hazards.enemies, model))
