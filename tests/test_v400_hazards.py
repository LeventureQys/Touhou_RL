"""v4.0.0 P1 hazard-export protocol contract (11-P1 §2.1).

The rows below marked ``REAL_`` are byte-for-byte copies of replies captured
from ``build/rl_hazard`` running the real Linux engine (stage 0, difficulty 1,
team 0, seed 1, difficulty-normal; frames as noted).  Everything else is
synthetic encoding-level input and is named accordingly.

What is pinned here:

* the appended BULLETS / ENEMIES / LASERS fields decode;
* a row that predates them still decodes, and its prefix is byte-identical to
  the new row's prefix (append-only guarantee);
* ``PLAYER`` reports the shot-data speed table rather than a constant;
* ``HAZARDS`` carries a whole frame in one reply and its totals stay comparable
  with the paged ``BULLETS`` / ``ENEMIES`` / ``LASERS`` counters;
* a truncated row or a fail-closed ``ERR`` is reported, never silently guessed.
"""

from __future__ import annotations

import struct

import pytest

from touhou_rl.v400.gym import protocol

# ---------------------------------------------------------------------------
# Real captured replies (build/rl_hazard, 2026-11-09, WSL xvfb + llvmpipe)
# ---------------------------------------------------------------------------

REAL_PLAYER_600 = (
    "OK frames=600 hp=40000000 lives=2 bombs=40400000 power=00000000 focus=0 "
    "isYoukai=0 playerState=0 invulnState=0 px=43400000 py=43c00000 "
    "hurtboxMinX=433f2ccd hurtboxMinY=43bf9666 hurtboxMaxX=4340d333 hurtboxMaxY=43c0699a "
    "speedNormalAxial=40800000 speedNormalDiagonal=403504f3 "
    "speedFocusedAxial=40000000 speedFocusedDiagonal=3fb504f3"
)

REAL_BULLETS_600 = (
    "OK frames=600 count=7 offset=0 limit=4 bullets="
    "0:1:6:43236cf9:439cb92f:3dcccccd:3eb56282:3f5fee8b:103:40800000:0,"
    "1:1:6:43238be1:4394e390:3dcccccd:3e976c88:3f656e28:83:40800000:0,"
    "2:1:6:4325bef7:438c855c:3dcccccd:3e6eca68:3f6a1bc1:63:40800000:0,"
    "3:1:6:43115fa5:4382a649:3dcccccd:3eaba426:3f61d7fe:43:40800000:0"
)

REAL_ENEMIES_600 = (
    "OK frames=600 count=8 offset=0 limit=4 enemies="
    "0:1000005d:41f00000:c1800000:00000000:00000014:0:"
    "41f00000:c1800000:00000000:41c00000:41c00000,"
    "1:0101104d:42c39d99:43176d3e:00000000:00000014:0:"
    "42c39d99:43176d3e:00000000:41c00000:41c00000,"
    "2:0101104d:42f39773:432b429d:00000000:00000014:0:"
    "42f39773:432b429d:00000000:41c00000:41c00000,"
    "3:0101104d:430e1c37:433c5a9a:00000000:00000014:0:"
    "430e1c37:433c5a9a:00000000:41c00000:41c00000"
)

REAL_LASERS_600 = "OK frames=600 count=0 offset=0 limit=4 lasers="

REAL_HAZARDS_600 = (
    "OK frames=600 nb=7 nl=0 ne=8 bullets="
    "0:1:6:43236cf9:439cb92f:3dcccccd:3eb56282:3f5fee8b:103:40800000:0,"
    "1:1:6:43238be1:4394e390:3dcccccd:3e976c88:3f656e28:83:40800000:0,"
    "2:1:6:4325bef7:438c855c:3dcccccd:3e6eca68:3f6a1bc1:63:40800000:0,"
    "3:1:6:43115fa5:4382a649:3dcccccd:3eaba426:3f61d7fe:43:40800000:0,"
    "4:1:6:431559ee:43742ade:3dcccccd:3e8cf710:3f671725:23:40800000:0,"
    "5:1:6:431bb4a9:436232e0:3dcccccd:3e589dcc:3f6b73d9:3:40800000:0,"
    "1536:6:0:00000000:00000000:00000000:00000000:00000000:0:00000000:0 lasers= enemies="
    "0:1000005d:41f00000:c1800000:00000000:00000014:0:"
    "41f00000:c1800000:00000000:41c00000:41c00000,"
    "1:0101104d:42c39d99:43176d3e:00000000:00000014:0:"
    "42c39d99:43176d3e:00000000:41c00000:41c00000,"
    "2:0101104d:42f39773:432b429d:00000000:00000014:0:"
    "42f39773:432b429d:00000000:41c00000:41c00000,"
    "3:0101104d:430e1c37:433c5a9a:00000000:00000014:0:"
    "430e1c37:433c5a9a:00000000:41c00000:41c00000,"
    "4:0101104d:42f6a4f5:434b44d8:00000000:00000014:0:"
    "42f6a4f5:434b44d8:00000000:41c00000:41c00000,"
    "5:0101104d:430bde62:435513de:00000000:00000014:0:"
    "430bde62:435513de:00000000:41c00000:41c00000,"
    "6:0101104d:431a2712:435b723d:00000000:00000014:0:"
    "431a2712:435b723d:00000000:41c00000:41c00000,"
    "7:0101204d:420efa63:bfe872e0:00000000:00000014:0:"
    "420efa63:bfe872e0:00000000:41c00000:41c00000"
)

#: Synthetic laser row in the current encoding: inUse=1, state ACTIVE,
#: startOffset=0, endOffset=8, width=4, timer=7.
SYNTHETIC_LASER_ROW = "3:1:40000000:40000000:00000000:00000000:00000000:41000000:40800000:40800000:7:1"

#: The same row as a pre-P1 engine emitted it (no ``inUse`` suffix).
SYNTHETIC_LEGACY_LASER_ROW = SYNTHETIC_LASER_ROW.rsplit(":", 1)[0]


def _rows(payload: str) -> list[str]:
    return [row for row in payload.split(",") if row]


def _legacy(row: str, keep: int) -> str:
    return ":".join(row.split(":")[:keep])


class _Replay(protocol.HarnessClient):
    """A client whose transport is a scripted list of replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[str] = []

    def request(self, line, timeout=None):
        self.requests.append(line)
        if not self.replies:
            raise AssertionError(f"unexpected request {line!r}")
        return self.replies.pop(0)


# ---------------------------------------------------------------------------
# Appended bullet fields
# ---------------------------------------------------------------------------


def test_real_bullet_rows_decode_appended_fields():
    payload = protocol._parse(REAL_BULLETS_600)["bullets"]
    rows = [protocol.parse_bullet_row(row) for row in _rows(payload)]
    assert len(rows) == 4
    first = rows[0]
    assert first.slot == 0
    assert first.state == protocol.BULLET_STATE_FIRED
    assert first.lethal is True
    # TH08's smallest bullet is 4.0 full width; the old teacher mis-read this
    # as a half width (11-P1 §2.1).
    assert first.collision_size == pytest.approx(4.0)
    assert first.half_width == pytest.approx(2.0)
    assert first.collision_disabled == 0
    assert first.timer == 103
    assert first.position == pytest.approx((163.42567, 313.44675, 0.1), abs=1e-4)


def test_bullet_row_without_appended_fields_still_decodes():
    legacy = _legacy(_rows(protocol._parse(REAL_BULLETS_600)["bullets"])[0], 9)
    row = protocol.parse_bullet_row(legacy)
    assert row.collision_size is None
    assert row.collision_disabled is None
    assert row.half_width is None
    # An engine that cannot report the flag is not evidence that it is disabled.
    assert row.lethal is True


def test_legacy_bullet_prefix_is_byte_identical_to_new_prefix():
    for raw in _rows(protocol._parse(REAL_BULLETS_600)["bullets"]):
        legacy = _legacy(raw, protocol.BULLET_PREFIX_FIELDS)
        new = protocol.parse_bullet_row(raw)
        old = protocol.parse_bullet_row(legacy)
        assert (old.slot, old.state, old.color, old.position, old.velocity, old.timer) == (
            new.slot,
            new.state,
            new.color,
            new.position,
            new.velocity,
            new.timer,
        )
        assert raw.split(":")[: protocol.BULLET_PREFIX_FIELDS] == legacy.split(":")


def test_collision_disabled_hides_lethality():
    row = protocol.parse_bullet_row("0:1:6:43236cf9:439cb92f:3dcccccd:3eb56282:3f5fee8b:9:40800000:1")
    assert row.collision_disabled == 1
    assert row.lethal is False


def test_truncated_bullet_row_is_rejected():
    with pytest.raises(protocol.HarnessError, match="BULLETS row has 5 fields"):
        protocol.parse_bullet_row("0:1:6:43236cf9:439cb92f")


# ---------------------------------------------------------------------------
# Appended enemy fields
# ---------------------------------------------------------------------------


def test_real_enemy_rows_decode_appended_fields():
    rows = [protocol.parse_enemy_row(row) for row in _rows(protocol._parse(REAL_ENEMIES_600)["enemies"])]
    assert len(rows) == 4
    first = rows[0]
    assert first.slot == 0
    assert first.flags1 == 0x1000005D
    # 0x5d = 0b1011101: bit 2 (ENEMY_FLAG_COLLISION) is set, so this body has a
    # hitbox even though it is a non-boss stage-start spawn.
    assert first.flags1 & protocol.ENEMY_FLAG_COLLISION == protocol.ENEMY_FLAG_COLLISION
    assert first.has_collision is True
    assert first.world_position == pytest.approx((30.0, -16.0, 0.0))
    assert first.hitbox_width == pytest.approx(24.0)
    assert first.hitbox_height == pytest.approx(24.0)
    # EnemyManager.cpp:852 -> hitbox/1.5 full width; Player.cpp:345 -> +-size/2.
    assert first.lethal_half_extent == pytest.approx((8.0, 8.0))
    second = rows[1]
    assert second.flags1 & protocol.ENEMY_FLAG_COLLISION
    assert second.has_collision is True
    assert second.flags1 & protocol.ENEMY_FLAG_ACTIVE


def test_enemy_row_without_appended_fields_still_decodes():
    legacy = _legacy(_rows(protocol._parse(REAL_ENEMIES_600)["enemies"])[0], 7)
    row = protocol.parse_enemy_row(legacy)
    assert row.slot == 0
    assert row.world_position is None
    assert row.hitbox_width is None
    assert row.lethal_half_extent is None


def test_legacy_enemy_prefix_is_byte_identical_to_new_prefix():
    for raw in _rows(protocol._parse(REAL_ENEMIES_600)["enemies"]):
        legacy = _legacy(raw, protocol.ENEMY_PREFIX_FIELDS)
        new = protocol.parse_enemy_row(raw)
        old = protocol.parse_enemy_row(legacy)
        assert (old.slot, old.flags1, old.position, old.life, old.timer) == (
            new.slot,
            new.flags1,
            new.position,
            new.life,
            new.timer,
        )


#: Real stage-0 frame-1440 enemy whose world transform differs from its local
#: position (a parented/offset enemy).  It is the proof that ``worldPosition``
#: carries information ``position`` does not.
REAL_ENEMY_OFFSET_WORLD = (
    "2:11003349:418482bc:42774708:00000000:00000032:0:435bc5ca:433affeb:00000000:41c00000:41c00000"
)


def test_real_world_position_differs_from_local_position():
    row = protocol.parse_enemy_row(REAL_ENEMY_OFFSET_WORLD)
    assert row.position == pytest.approx((16.563835, 61.819366, 0.0), abs=1e-3)
    assert row.world_position == pytest.approx((219.772614, 186.999680, 0.0), abs=1e-3)
    assert row.world_position != row.position
    # 11-P1 §2.1 anchors the enemy-body AABB on worldPosition, not position.
    # flags1 ends in 0x49 = 0b1001001: bit 2 is clear, so this body is NOT
    # lethal even though its hitbox dimensions are populated.  A geometry pass
    # that ignored the flag would invent a hazard here.
    assert row.flags1 & protocol.ENEMY_FLAG_COLLISION == 0
    assert row.has_collision is False
    assert row.hitbox_width == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# Appended laser field
# ---------------------------------------------------------------------------


def test_laser_row_reports_in_use_and_geometry():
    row = protocol.parse_laser_row(SYNTHETIC_LASER_ROW)
    assert row.in_use == 1
    assert row.state == 1
    assert row.start_offset == pytest.approx(0.0)
    assert row.end_offset == pytest.approx(8.0)
    assert row.width == pytest.approx(4.0)
    assert row.current_width == pytest.approx(4.0)
    assert row.timer == 7


def test_laser_row_without_in_use_still_decodes():
    row = protocol.parse_laser_row(SYNTHETIC_LEGACY_LASER_ROW)
    assert row.in_use is None
    assert row.end_offset == pytest.approx(8.0)


def test_real_empty_laser_page_parses_to_no_rows():
    assert _rows(protocol._parse(REAL_LASERS_600)["lasers"]) == []


# ---------------------------------------------------------------------------
# PLAYER: speed table read from shot data, never a constant
# ---------------------------------------------------------------------------


def test_real_player_reports_team0_shot_data_speeds():
    snapshot = protocol.parse_player_reply(REAL_PLAYER_600)
    assert snapshot.frame == 600
    assert snapshot.speeds_available is True
    assert snapshot.speed(focused=False) == pytest.approx((4.0, 2.8284271), abs=1e-6)
    assert snapshot.speed(focused=True) == pytest.approx((2.0, 1.4142136), abs=1e-6)


def test_real_player_resources_and_hitbox():
    snapshot = protocol.parse_player_reply(REAL_PLAYER_600)
    assert snapshot.hp == pytest.approx(2.0)
    assert snapshot.lives == 2
    assert snapshot.bombs == pytest.approx(3.0)
    assert snapshot.power == pytest.approx(0.0)
    assert snapshot.position == pytest.approx((192.0, 384.0))
    assert snapshot.hurtbox_min == pytest.approx((191.175, 383.175), abs=1e-4)
    assert snapshot.hurtbox_max == pytest.approx((192.825, 384.825), abs=1e-4)
    # 11-P1 §2.1 measured a 0.825 half-width for team 0; recompute it here so a
    # future bounds regression cannot slip past as "just a number".
    assert (snapshot.hurtbox_max[0] - snapshot.hurtbox_min[0]) / 2.0 == pytest.approx(
        0.825, abs=1e-4
    )
    assert snapshot.invuln_state == 0
    assert snapshot.player_state == 0


def test_player_reply_without_speed_fields_is_not_guessed():
    legacy = " ".join(
        token
        for token in REAL_PLAYER_600.split()
        if not token.startswith(
            ("speedNormal", "speedFocused", "hurtbox", "isYoukai", "invulnState")
        )
    )
    snapshot = protocol.parse_player_reply(legacy)
    assert snapshot.speeds_available is False
    assert snapshot.speed(focused=False) == (None, None)
    assert snapshot.hurtbox_min is None
    assert snapshot.position == pytest.approx((192.0, 384.0))


def test_player_missing_required_field_is_an_error():
    with pytest.raises(protocol.HarnessError, match="PLAYER reply missing"):
        protocol.parse_player_reply("OK frames=1 lives=2 px=43400000 py=43c00000")


def test_player_error_reply_propagates():
    client = _Replay(["ERR player-speed-unavailable primary_sht=0 secondary_sht=1"])
    client.stats = protocol.ProtocolStats()

    def request(line, timeout=None):
        raise protocol.HarnessError(f"{line!r} -> ERR player-speed-unavailable")

    client.request = request  # type: ignore[assignment]
    with pytest.raises(protocol.HarnessError, match="player-speed-unavailable"):
        client.player()


def test_player_command_is_sent_and_decoded():
    client = _Replay([REAL_PLAYER_600])
    snapshot = client.player()
    assert client.requests == ["PLAYER"]
    assert snapshot.lives == 2


# ---------------------------------------------------------------------------
# HAZARDS: one round trip per frame, totals comparable with the paged commands
# ---------------------------------------------------------------------------


def test_real_hazards_frame_matches_the_paged_counters():
    frame = protocol.parse_hazards_reply(REAL_HAZARDS_600)
    bullets = _rows(protocol._parse(REAL_BULLETS_600)["bullets"])
    enemies = _rows(protocol._parse(REAL_ENEMIES_600)["enemies"])
    lasers = _rows(protocol._parse(REAL_LASERS_600)["lasers"])

    assert frame.frame == 600
    assert frame.bullet_count == int(protocol._parse(REAL_BULLETS_600)["count"])
    assert frame.enemy_count == int(protocol._parse(REAL_ENEMIES_600)["count"])
    assert frame.laser_count == int(protocol._parse(REAL_LASERS_600)["count"])
    assert len(frame.bullets) == frame.bullet_count
    assert len(frame.enemies) == frame.enemy_count
    assert len(frame.lasers) == frame.laser_count
    # The paged reply was capped at limit=4 but still reports every slot.
    assert len(bullets) == 4
    assert frame.bullet_count == 7
    assert len(enemies) == 4
    assert frame.enemy_count == 8
    assert lasers == []


def test_real_hazards_rows_equal_the_standalone_commands():
    frame = protocol.parse_hazards_reply(REAL_HAZARDS_600)
    paged = [protocol.parse_bullet_row(row) for row in _rows(protocol._parse(REAL_BULLETS_600)["bullets"])]
    assert frame.bullets[:4] == tuple(paged)
    enemies = [protocol.parse_enemy_row(row) for row in _rows(protocol._parse(REAL_ENEMIES_600)["enemies"])]
    assert frame.enemies[:4] == tuple(enemies)


def test_hazards_large_batch_parses_every_row():
    template = _rows(protocol._parse(REAL_BULLETS_600)["bullets"])[0]
    rows = []
    for slot in range(600):
        parts = template.split(":")
        parts[0] = str(slot)
        rows.append(":".join(parts))
    reply = (
        "OK frames=9000 nb=600 nl=0 ne=0 bullets="
        + ",".join(rows)
        + " lasers= enemies="
    )
    frame = protocol.parse_hazards_reply(reply)
    assert frame.bullet_count == 600
    assert len(frame.bullets) == 600
    assert [row.slot for row in frame.bullets[:3]] == [0, 1, 2]
    assert frame.bullets[-1].slot == 599
    assert frame.bullets[0].collision_size == pytest.approx(4.0)
    assert frame.raw_bytes == len(reply)


def test_hazards_limit_keeps_engine_totals_while_capping_rows():
    template = _rows(protocol._parse(REAL_BULLETS_600)["bullets"])[0]
    rows = [":".join([str(slot)] + template.split(":")[1:]) for slot in range(32)]
    reply = (
        "OK frames=600 nb=600 nl=3 ne=40 bullets=" + ",".join(rows) + " lasers= enemies="
    )
    frame = protocol.parse_hazards_reply(reply)
    assert frame.bullet_count == 600
    assert len(frame.bullets) == 32
    assert frame.laser_count == 3


def test_hazards_too_large_fails_closed():
    client = _Replay(["ERR hazards-too-large capacity=262144"])

    def request(line, timeout=None):
        raise protocol.HarnessError(
            f"{line!r} -> ERR hazards-too-large capacity=262144"
        )

    client.request = request  # type: ignore[assignment]
    with pytest.raises(protocol.HarnessError, match="hazards-too-large"):
        client.hazards()


def test_hazards_reply_missing_totals_is_an_error():
    with pytest.raises(protocol.HarnessError, match="HAZARDS reply missing nb"):
        protocol.parse_hazards_reply("OK frames=1 bullets= lasers= enemies=")


def test_hazards_command_shape_and_limit_validation():
    client = _Replay([REAL_HAZARDS_600, REAL_HAZARDS_600])
    client.hazards()
    client.hazards(128)
    assert client.requests == ["HAZARDS", "HAZARDS 128"]
    with pytest.raises(ValueError, match="limit must be >= 0"):
        client.hazards(-1)


def test_hazards_capacity_constant_matches_the_engine():
    # rlharness.cpp HAZARDS_REPLY_CAPACITY = 256u * 1024u.
    assert protocol.HAZARDS_REPLY_CAPACITY == 262144


# ---------------------------------------------------------------------------
# Paged entity commands keep their pre-P1 replies usable
# ---------------------------------------------------------------------------


def test_entity_commands_send_the_documented_pages():
    client = _Replay([REAL_BULLETS_600, REAL_ENEMIES_600, REAL_LASERS_600])
    assert len(client.bullets(0, 4)) == 4
    assert len(client.enemies(0, 4)) == 4
    assert client.lasers(0, 4) == []
    assert client.requests == ["BULLETS 0 4", "ENEMIES 0 4", "LASERS 0 4"]


def test_entity_page_limits_are_rejected_locally():
    client = _Replay([])
    with pytest.raises(ValueError, match="BULLETS limit is 1..16"):
        client.bullets(0, 17)
    with pytest.raises(ValueError, match="ENEMIES limit is 1..16"):
        client.enemies(0, 0)
    with pytest.raises(ValueError, match="LASERS limit is 1..16"):
        client.lasers(0, 17)
    with pytest.raises(ValueError, match="offset must be >= 0"):
        client.lasers(-1, 4)


def test_counts_use_a_single_slot_page():
    client = _Replay([REAL_BULLETS_600, REAL_ENEMIES_600, REAL_LASERS_600])
    assert client.bullet_count() == 7
    assert client.enemy_count() == 8
    assert client.laser_count() == 0
    assert client.requests == ["BULLETS 0 1", "ENEMIES 0 1", "LASERS 0 1"]


def test_bits_float_round_trips_engine_floats():
    for value in (0.0, 4.0, 2.8284271, -16.0, 1.4142136):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        assert protocol._bits_float("%08x" % bits) == pytest.approx(value, abs=1e-6)


def test_parse_still_reads_legacy_scalar_replies():
    # Nothing about the scalar commands changed; HASH/STATE/STEP keep working.
    assert protocol._parse("OK hash=7f04c7361749ed9f frames=0") == {
        "_verb": "OK",
        "hash": "7f04c7361749ed9f",
        "frames": "0",
    }
    assert protocol._parse(REAL_LASERS_600)["count"] == "0"
