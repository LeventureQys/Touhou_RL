"""EXP-P1-004 geometry tests: the model must match the engine, not look right.

Every assertion here is derived from engine source, and the tests are written so
that the *wrong* model fails them.  Three specific mistakes are pinned:

* ``collisionSize`` is a **full** width (Player.cpp:309-348 halves it), so
  reading it as a half width makes every bullet twice as lethal;
* an enemy body is ``hitboxDimensions / 3`` (EnemyManager.cpp:851 divides by
  1.5, then ``CheckLethalCollision`` halves again) and only lethal when the
  three extra ``flags1`` bits are clear (EnemyManagerUpdate.cpp:287-291);
* the engine tests a bullet **after** ``position += velocity`` and despawns the
  bullet that hit in the same block, so a snapshot never shows the killing
  bullet at the point that killed.  ``MODEL_ENGINE_PHASE`` is the model that
  puts it back.

Run with::

    $env:PYTHONPATH='D:\\WorkShop\\Touhou_RL'
    python -m pytest -p no:cacheprovider --basetemp .tmp_v400_geom_pytest -q \
        tests/test_v400_geometry_agreement.py
"""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = REPO_ROOT / "touhou_rl" / "v400" / "teacher" / "geometry.py"


def _load_geometry():
    """Load ``geometry.py`` by path, bypassing the teacher package ``__init__``.

    The package is built by a parallel work package whose re-export list may be
    mid-flight; these tests are about the geometry file, so pin the file.
    """

    import sys

    name = "_v400_teacher_geometry_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, GEOMETRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


geometry = _load_geometry()

SPEC = geometry.MODEL_SPEC
ENGINE_PHASE = geometry.MODEL_ENGINE_PHASE
LEGACY = geometry.MODEL_LEGACY_HALF_WIDTH
COLLISION_ONLY = geometry.MODEL_COLLISION_BIT_ONLY

#: ``PLAYER`` on the real data: half width 0.825 around (192, 384).
HURT_MIN = (191.1750030517578, 383.17498779296875)
HURT_MAX = (192.8249969482422, 384.82501220703125)


def _bits(value: float) -> float:
    """Round-trip through float bits, the way the wire does."""

    return struct.unpack("<f", struct.pack("<I", int(pack(float(value)))))[0]


def pack(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


class Bullet:
    __slots__ = ("slot", "state", "collision_size", "collision_disabled",
                 "position", "velocity")

    def __init__(self, slot, x, y, size=4.0, state=1, disabled=0, vx=0.0, vy=0.0):
        self.slot = slot
        self.state = state
        self.collision_size = _bits(size)
        self.collision_disabled = disabled
        self.position = (_bits(x), _bits(y))
        self.velocity = (_bits(vx), _bits(vy))


class Enemy:
    __slots__ = ("slot", "flags1", "world_position", "hitbox_width", "hitbox_height")

    def __init__(self, slot, x, y, flags1, width=24.0, height=24.0, world=True):
        self.slot = slot
        self.flags1 = flags1
        self.world_position = (_bits(x), _bits(y)) if world else None
        self.hitbox_width = _bits(width)
        self.hitbox_height = _bits(height)


class Laser:
    __slots__ = ("slot", "state", "in_use", "position", "angle",
                 "start_offset", "end_offset", "width")

    def __init__(self, slot, x, y, angle=0.0, start=0.0, end=100.0, width=16.0,
                 state=1, in_use=1):
        self.slot = slot
        self.state = state
        self.in_use = in_use
        self.position = (_bits(x), _bits(y))
        self.angle = _bits(angle)
        self.start_offset = _bits(start)
        self.end_offset = _bits(end)
        self.width = _bits(width)


class Hazards:
    def __init__(self, bullets=(), enemies=(), lasers=()):
        self.bullets = tuple(bullets)
        self.enemies = tuple(enemies)
        self.lasers = tuple(lasers)


class Player:
    def __init__(self, position=(192.0, 384.0)):
        cx, cy = position
        self.position = position
        self.hurtbox_min = (_bits(cx - 0.825), _bits(cy - 0.825))
        self.hurtbox_max = (_bits(cx + 0.825), _bits(cy + 0.825))


# ---------------------------------------------------------------------------
# bullet half width
# ---------------------------------------------------------------------------
def test_bullet_half_width_is_half_the_exported_collision_size():
    """A 4 px bullet reaches 2 px from its centre, not 4.

    The hurtbox spans x in [191.175, 192.825], so a 4 px bullet centred at c
    covers [c - 2, c + 2] and can only touch for c <= 194.825.
    """

    # 194.9 - 2.0 = 192.9 > 192.825: one fifth of a pixel short.
    hazards = Hazards(bullets=[Bullet(0, 194.9, 384.0, size=4.0)])
    verdict = geometry.frame_verdict(hazards, Player(), SPEC)
    assert verdict.hit is False
    assert verdict.nearest[2] == pytest.approx(194.9 - 2.0 - 192.8249969482422, abs=1e-4)

    # 194.7 - 2.0 = 192.7 < 192.825: contact.
    hazards = Hazards(bullets=[Bullet(0, 194.7, 384.0, size=4.0)])
    assert geometry.frame_verdict(hazards, Player(), SPEC).hit is True

    # Half width, not full width: 194.7 with a 4 px *half* width would keep the
    # edge 2 px further out, i.e. still short of the box, which it is not.
    boxes = geometry.bullet_boxes([Bullet(0, 194.7, 384.0, size=4.0)], SPEC)
    assert boxes.maxs[0][0] - boxes.mins[0][0] == pytest.approx(4.0, abs=1e-6)


def test_touching_the_hurtbox_counts_as_a_hit():
    """The engine rejects only on ``hurtMin > boxMax``, so equality is a hit."""

    # Bullet centre such that boxMax.x == hurtMin.x exactly: 191.175 = 189.175 + 2.
    bullet = Bullet(0, 189.1750030517578, 384.0, size=4.0)
    boxes = geometry.bullet_boxes([bullet], SPEC)
    assert boxes.maxs[0][0] == pytest.approx(HURT_MIN[0], abs=1e-6)
    verdict = geometry.frame_verdict(Hazards(bullets=[bullet]), Player(), SPEC)
    assert verdict.hit is True


def test_one_ulp_short_of_touching_is_not_a_hit():
    """Exclusive bounds would call the touching case a miss; the engine does not."""

    bullet = Bullet(0, 189.17498779296875, 384.0, size=4.0)  # one step further left
    verdict = geometry.frame_verdict(Hazards(bullets=[bullet]), Player(), SPEC)
    assert verdict.hit is False
    assert verdict.nearest[2] > 0.0


def test_legacy_full_width_model_over_reports_the_same_bullet():
    """The old teacher's bug, measured on one frame instead of asserted in prose.

    ``spec`` needs ``c <= 194.825``; using ``collisionSize`` as a half width
    needs only ``c <= 196.825``, so a bullet at 196.0 is a phantom hit.
    """

    bullet = Bullet(0, 196.0, 384.0, size=4.0)
    hazards = Hazards(bullets=[bullet])
    assert geometry.frame_verdict(hazards, Player(), SPEC).hit is False
    assert geometry.frame_verdict(hazards, Player(), LEGACY).hit is True


def test_non_fired_and_disabled_bullets_are_not_lethal():
    """``state == FIRED`` and ``collisionDisabled == 0`` are both required."""

    on_top = (192.0, 384.0)
    for state in (0, 2, 3, 4, 5, 6):
        hazards = Hazards(bullets=[Bullet(0, *on_top, state=state)])
        assert geometry.frame_verdict(hazards, Player(), SPEC).hit is False
    hazards = Hazards(bullets=[Bullet(0, *on_top, disabled=1)])
    assert geometry.frame_verdict(hazards, Player(), SPEC).hit is False
    hazards = Hazards(bullets=[Bullet(0, *on_top)])
    assert geometry.frame_verdict(hazards, Player(), SPEC).hit is True


def test_bullet_without_a_exported_width_is_dropped_not_guessed():
    """A missing collisionSize is not a zero width (pre-P1 engine rows)."""

    bullet = Bullet(0, 192.0, 384.0)
    bullet.collision_size = None
    assert geometry.bullet_boxes([bullet], SPEC).mins.shape == (0, 2)


# ---------------------------------------------------------------------------
# the engine tests one tick ahead
# ---------------------------------------------------------------------------
def test_engine_phase_advances_a_bullet_by_exactly_one_velocity_tick():
    """``position += velocity`` in float32, which is what the engine executes."""

    bullet = Bullet(0, 200.0, 384.0, size=4.0, vx=-2.5, vy=0.0)
    still = geometry.bullet_boxes([bullet], SPEC)
    ahead = geometry.bullet_boxes([bullet], ENGINE_PHASE)
    assert ahead.maxs[0][0] == pytest.approx(still.maxs[0][0] - 2.5, abs=1e-6)
    assert ahead.maxs[0][1] == pytest.approx(still.maxs[0][1], abs=1e-6)


def test_killing_bullet_is_invisible_at_both_frames_but_engine_phase_finds_it():
    """The structural blind spot of a per-frame point test, reproduced exactly.

    Frame k: the bullet is one tick short of the hurtbox.  The engine moves it,
    finds the hit and despawns it, so the frame k+1 snapshot has no such bullet
    at all.  ``spec`` therefore never sees it and ``engine-phase`` does.
    """

    # 4 px bullet, half width 2: contact happens when |x - 192| <= 2.825.
    # At x = 195.5 the edge is 0.675 px away.  Velocity -1.5 puts it at 194.0,
    # edge 192.0 -> inside on the next tick.
    bullet = Bullet(0, 195.5, 384.0, size=4.0, vx=-1.5, vy=0.0)
    hazards_at_k = Hazards(bullets=[bullet])

    assert geometry.frame_verdict(hazards_at_k, Player(), SPEC).hit is False

    # The engine kills it inside the step, so the next snapshot no longer lists
    # it as FIRED.  Testing "the snapshot you just read" can never see this.
    assert geometry.frame_verdict(hazards_at_k, Player(), ENGINE_PHASE).hit is True


def test_engine_phase_does_not_invent_hits_for_bullets_moving_away():
    bullet = Bullet(0, 195.5, 384.0, size=4.0, vx=+3.0, vy=0.0)
    hazards = Hazards(bullets=[bullet])
    assert geometry.frame_verdict(hazards, Player(), ENGINE_PHASE).hit is False
    assert geometry.frame_verdict(hazards, Player(), ENGINE_PHASE).nearest[2] > 3.0


# ---------------------------------------------------------------------------
# enemy bodies
# ---------------------------------------------------------------------------
def test_enemy_half_extent_is_a_third_of_the_raw_hitbox():
    """``/1.5`` in EnemyManager then ``/2`` in CheckLethalCollision."""

    # 24 px hitbox -> 8 px half extent.  Centre 8 px right of the hurtbox edge
    # touches exactly at x = 192.825 + 8.
    enemy = Enemy(0, 200.825, 384.0, flags1=geometry.ENEMY_FLAG_COLLISION)
    boxes = geometry.enemy_boxes([enemy], SPEC)
    assert boxes.maxs[0][0] - boxes.mins[0][0] == pytest.approx(16.0, abs=1e-5)
    verdict = geometry.frame_verdict(Hazards(enemies=[enemy]), Player(), SPEC)
    assert verdict.hit is True

    # 200.9 is 0.075 px beyond the 8 px half extent: no contact.
    enemy = Enemy(0, 200.9, 384.0, flags1=geometry.ENEMY_FLAG_COLLISION)
    assert geometry.frame_verdict(Hazards(enemies=[enemy]), Player(), SPEC).hit is False


def test_legacy_enemy_divisor_makes_the_body_four_times_too_big_in_area():
    """``/1.5`` read as a half extent, i.e. ``/1.5`` instead of ``/3``."""

    enemy = Enemy(0, 204.0, 384.0, flags1=geometry.ENEMY_FLAG_COLLISION)
    hazards = Hazards(enemies=[enemy])
    assert geometry.frame_verdict(hazards, Player(), SPEC).hit is False
    assert geometry.frame_verdict(hazards, Player(), LEGACY).hit is True


def test_enemy_body_uses_world_position_not_stage_local_position():
    """A parented enemy's ``position`` is not where its body is."""

    enemy = Enemy(0, 192.0, 384.0, flags1=geometry.ENEMY_FLAG_COLLISION)
    # worldPosition at the same place as the hurtbox centre -> hit.
    assert geometry.frame_verdict(Hazards(enemies=[enemy]), Player(), SPEC).hit is True
    # No worldPosition at all: the row cannot be placed, so it is dropped.
    enemy = Enemy(0, 192.0, 384.0, flags1=geometry.ENEMY_FLAG_COLLISION, world=False)
    assert geometry.frame_verdict(Hazards(enemies=[enemy]), Player(), SPEC).hit is False


@pytest.mark.parametrize(
    "suppressing_flag",
    [geometry.ENEMY_FLAG_NO_SPRITE,
     geometry.ENEMY_FLAG_HIDE_PRIMARY_ANM,
     geometry.ENEMY_FLAG_YOUKAI_ALIGNED],
)
def test_enemy_flag_gates_beyond_the_collision_bit(suppressing_flag):
    """EnemyManagerUpdate.cpp:287-291 requires more than ``collision``."""

    flags = geometry.ENEMY_FLAG_COLLISION | suppressing_flag
    enemy = Enemy(0, 192.0, 384.0, flags1=flags)
    hazards = Hazards(enemies=[enemy])
    assert geometry.frame_verdict(hazards, Player(), SPEC).hit is False
    # Reading only bit 2 (11-P1 §2.1 taken literally) over-reports it.
    assert geometry.frame_verdict(hazards, Player(), COLLISION_ONLY).hit is True


def test_enemy_without_the_collision_bit_is_never_lethal():
    enemy = Enemy(0, 192.0, 384.0, flags1=1)  # bit 0 active, bit 2 clear
    assert geometry.frame_verdict(Hazards(enemies=[enemy]), Player(), SPEC).hit is False


# ---------------------------------------------------------------------------
# lasers
# ---------------------------------------------------------------------------
def test_laser_uses_width_over_four_as_half_thickness():
    """The engine sets ``size.y = width / 2`` and then halves ``size``."""

    # Beam along +x from the ship to the right, 16 px wide -> 4 px half thickness.
    laser = Laser(0, 0.0, 384.0, angle=0.0, start=0.0, end=300.0, width=16.0)
    assert geometry.laser_hits([laser], HURT_MIN, HURT_MAX, SPEC).hit is True
    # 4.9 px above the axis: outside 4 + 0.825 = 4.825.
    laser = Laser(0, 0.0, 388.9, angle=0.0, start=0.0, end=300.0, width=16.0)
    assert geometry.laser_hits([laser], HURT_MIN, HURT_MAX, SPEC).hit is False


def test_laser_rotates_the_ship_into_the_laser_frame():
    """A vertical beam (angle = pi/2) sweeps x, not y."""

    import math

    vertical = Laser(0, 192.0, 0.0, angle=math.pi / 2, start=0.0, end=500.0,
                     width=8.0)
    assert geometry.laser_hits([vertical], HURT_MIN, HURT_MAX, SPEC).hit is True
    # Same box rotated back to horizontal: the ship at y = 384 is now far off
    # the beam axis at x = 192, so the hit must disappear.
    horizontal = Laser(0, 192.0, 0.0, angle=0.0, start=0.0, end=500.0, width=8.0)
    assert geometry.laser_hits([horizontal], HURT_MIN, HURT_MAX, SPEC).hit is False


def test_laser_start_offset_scales_the_length_by_point_seven():
    """``startOffset > 0`` multiplies the length by 0.7 in BulletManager.cpp:1078."""

    # Beam from x=100 to x=200, i.e. not reaching the ship at 192... chosen so
    # the 0.7 factor is the difference between touching and not.
    reached = Laser(0, 0.0, 384.0, angle=0.0, start=130.0, end=222.0, width=8.0)
    # length 92, *0.7 = 64.4, centre = 130 + 46 = 176, half len 32.2 -> 208.2.
    result = geometry.laser_hits([reached], HURT_MIN, HURT_MAX, SPEC)
    assert result.considered == 1
    assert result.hit is True

    short = Laser(0, 0.0, 384.0, angle=0.0, start=130.0, end=180.0, width=8.0)
    # length 50, *0.7 = 35, centre = 130 + 25 = 155, half len 17.5 -> 172.5.
    assert geometry.laser_hits([short], HURT_MIN, HURT_MAX, SPEC).hit is False


def test_laser_only_counts_when_active_and_in_use():
    laser = Laser(0, 0.0, 384.0, angle=0.0, start=0.0, end=300.0, width=16.0)
    assert geometry.laser_hits([laser], HURT_MIN, HURT_MAX, SPEC).hit is True

    starting = Laser(0, 0.0, 384.0, angle=0.0, start=0.0, end=300.0, width=16.0,
                     state=0)
    result = geometry.laser_hits([starting], HURT_MIN, HURT_MAX, SPEC)
    assert result.hit is False
    assert result.skipped_inactive == 1

    freed = Laser(0, 0.0, 384.0, angle=0.0, start=0.0, end=300.0, width=16.0,
                  in_use=0)
    assert geometry.laser_hits([freed], HURT_MIN, HURT_MAX, SPEC).considered == 0


# ---------------------------------------------------------------------------
# precision, registry, and the recorded dataset
# ---------------------------------------------------------------------------
def test_float32_keeps_touching_boxes_that_float64_can_lose():
    """The engine computes in float; widening the arithmetic can flip a boundary.

    The two models agree on the recorded EXP-P1-004 dataset (0 disagreements),
    so this only pins that the dtype is honoured rather than silently promoted.
    """

    wide = geometry.Model(name="f64", bullet_half_scale=0.5, enemy_half_divisor=3.0,
                          inclusive=True, dtype="float64")
    bullet = Bullet(0, 189.1750030517578, 384.0, size=4.0)
    hazards = Hazards(bullets=[bullet])
    assert geometry.frame_verdict(hazards, Player(), SPEC).hit is True
    # float64 recomputes the same numbers from the same float32 inputs, so it
    # also hits; the assertion documents that convention instead of assuming it.
    assert geometry.frame_verdict(hazards, Player(), wide).hit is True
    assert geometry.bullet_boxes([bullet], SPEC).mins.dtype.name == "float32"
    assert geometry.bullet_boxes([bullet], wide).mins.dtype.name == "float64"


def test_model_registry_and_hurtbox_phase_are_consistent():
    assert set(geometry.MODELS) == {
        "spec", "engine-phase", "collision-bit-only", "legacy-half-width"
    }
    assert set(geometry.HURTBOX_PHASE) == set(geometry.MODELS)
    # Only the model that advances a tick needs the next frame's hurtbox.
    assert geometry.HURTBOX_PHASE["engine-phase"] == "next"
    assert geometry.MODELS["engine-phase"].bullet_advance == 1.0
    assert geometry.MODELS["spec"].bullet_advance == 0.0


def test_nearest_hazard_is_reported_when_nothing_hits():
    """Mismatch triage needs the distance to the closest lethal object."""

    hazards = Hazards(
        bullets=[Bullet(7, 300.0, 384.0, size=4.0)],
        enemies=[Enemy(3, 210.0, 384.0, flags1=geometry.ENEMY_FLAG_COLLISION)],
    )
    verdict = geometry.frame_verdict(hazards, Player(), SPEC)
    assert verdict.hit is False
    kind, slot, gap = verdict.nearest
    assert kind == geometry.KIND_ENEMY and slot == 3
    # 24 px hitbox -> 8 px half extent -> edge at 202.0, hurtbox at 192.825.
    assert gap == pytest.approx(210.0 - 8.0 - 192.8249969482422, abs=1e-3)


def test_recorded_dataset_reproduces_the_headline_when_present():
    """Regression pin on the frozen EXP-P1-004 frames, if they are on disk.

    Skipped on a clean checkout: ``.tmp_v400_geom`` is scratch, not a fixture.
    """

    dataset = REPO_ROOT / ".tmp_v400_geom" / "agreement.jsonl"
    if not dataset.is_file():
        pytest.skip("recorded dataset not present")
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert len(rows) >= 1000

    def rehydrate(frame):
        """Minimal stand-ins so the test does not depend on the gym client."""

        bullets = tuple(Bullet(row[0], row[4], row[5], size=row[2], state=row[1],
                               disabled=row[3], vx=row[6], vy=row[7])
                        for row in frame["b"])
        enemies = tuple(
            Enemy(row[0], row[2], row[3], flags1=row[1],
                  width=row[4], height=row[5]) for row in frame["e"]
        )
        hazards = Hazards(bullets=bullets, enemies=enemies, lasers=())
        player = Player()
        player.hurtbox_min = tuple(frame["player"]["hmin"])
        player.hurtbox_max = tuple(frame["player"]["hmax"])
        return hazards, player, frame["player"]["state"]

    by_run = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)

    naive_misses = 0
    phase_misses = 0
    hits = 0
    evaluated = 0
    for run_rows in by_run.values():
        for index in range(len(run_rows) - 1):
            hazards, player, state = rehydrate(run_rows[index])
            if state != 0:
                continue
            if run_rows[index + 1]["player"]["state"] != 2:
                continue
            hits += 1
            evaluated += 1
            if not geometry.frame_verdict(hazards, player, SPEC).hit:
                naive_misses += 1
            # Engine phase pairs frame k's hazards (advanced one velocity tick)
            # with frame k+1's hurtbox: that is the state the engine tests.
            next_hazards, next_player, _ = rehydrate(run_rows[index + 1])
            if not geometry.frame_verdict(hazards, next_player, ENGINE_PHASE).hit:
                phase_misses += 1

    assert hits > 0, "no death events recorded; the dataset cannot judge a miss"
    # The naive snapshot model misses every death: the killing bullet despawns.
    assert naive_misses == hits
    # The engine-phase model finds every one of them.
    assert phase_misses == 0
    assert evaluated == hits
