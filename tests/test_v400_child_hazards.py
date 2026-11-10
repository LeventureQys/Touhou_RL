"""EXP-P1-005: ``EVENTS`` and read-only ``CHILD`` snapshots.

Two layers:

* **Encoding** — the fixtures marked ``REAL_`` are byte-for-byte replies captured
  from ``build/rl_hazard2`` running the real Linux engine (stage 0, difficulty 1,
  team 0, seed 1, shadow role, 2026-11-09).  They pin the decoders, the
  ``CHILD`` command formatting and the fail-closed rules.
* **Live** — launches the real ``build/rl_hazard2`` through WSL and checks on an
  actual fork that ``CHILD <h> HAZARDS`` / ``PLAYER`` / ``EVENTS`` answer with
  the parent's semantics, that a child step does not advance the parent, and
  that a >4 KiB HAZARDS reply survives the forwarding path.  Skipped when the
  binary, the calibration data or WSL is unavailable.

``EVENTS`` is deliberately separate from the 25 tracked ``HASH``/``STATE``
fields; no test here expects an event flag inside ``STATE``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from touhou_rl.v400.gym import protocol

# ---------------------------------------------------------------------------
# Real captured replies (build/rl_hazard2, 2026-11-09, WSL, seed 1)
# ---------------------------------------------------------------------------

# ``RESET 0 1 0 1`` then ``STEP 0 120`` (frame 120).
REAL_HAZARDS_120 = (
    "OK frames=120 nb=1 nl=0 ne=1 bullets="
    "1536:6:0:00000000:00000000:00000000:00000000:00000000:0:00000000:0 "
    "lasers= "
    "enemies=0:1000005d:41f00000:c1800000:00000000:00000014:0:"
    "41f00000:c1800000:00000000:41c00000:41c00000"
)
REAL_PLAYER_120 = (
    "OK frames=120 hp=40000000 lives=2 bombs=40400000 power=00000000 focus=0 "
    "isYoukai=0 playerState=3 invulnState=1 px=43400000 py=43c00000 "
    "hurtboxMinX=433f2ccd hurtboxMinY=43bf9666 "
    "hurtboxMaxX=4340d333 hurtboxMaxY=43c0699a "
    "speedNormalAxial=40800000 speedNormalDiagonal=403504f3 "
    "speedFocusedAxial=40000000 speedFocusedDiagonal=3fb504f3"
)
REAL_CHILD_BULLETS_120 = (
    "OK frames=120 count=1 offset=0 limit=4 bullets="
    "1536:6:0:00000000:00000000:00000000:00000000:00000000:0:00000000:0"
)
REAL_CHILD_ENEMIES_120 = (
    "OK frames=120 count=1 offset=0 limit=4 enemies="
    "0:1000005d:41f00000:c1800000:00000000:00000014:0:"
    "41f00000:c1800000:00000000:41c00000:41c00000"
)
REAL_CHILD_LASERS_120 = "OK frames=120 count=0 offset=0 limit=4 lasers="
# ``EVENTS`` before a stage has a ReplayManager, and after one has one.
REAL_EVENTS_UNAVAILABLE = "ERR events-unavailable g_replaymanager=0"
REAL_EVENTS_0000 = "OK frames=120 frameEventFlags=0000"
# First idle-policy player hit on the real engine, observed step 678 / frame 678.
# 0x0602 = PLAYER_HIT (0x2) | PLAYER_DYING_STARTED (0x200) | EFFECT_SPAWNED (0x400).
REAL_EVENTS_PLAYER_HIT_678 = "OK frames=678 frameEventFlags=0602"

# ---------------------------------------------------------------------------
# Synthetic encoding-level input (not engine evidence)
# ---------------------------------------------------------------------------

SYNTHETIC_LARGE_HAZARDS = "OK frames=9 nb=64 nl=0 ne=0 bullets=" + ",".join(
    f"{slot}:1:2:00000000:00000000:3dcccccd:00000000:00000000:1:40800000:0"
    for slot in range(64)
) + " lasers= enemies="


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
# EVENTS decoding
# ---------------------------------------------------------------------------


def test_events_reply_decodes_the_player_hit_bit():
    events = protocol.parse_events_reply(REAL_EVENTS_PLAYER_HIT_678)

    assert events.frame == 678
    assert events.frame_event_flags == 0x0602
    assert events.player_hit is True
    assert events.names == ("PLAYER_HIT", "PLAYER_DYING_STARTED", "EFFECT_SPAWNED")


def test_events_reply_without_the_bit_is_not_a_hit():
    events = protocol.parse_events_reply(REAL_EVENTS_0000)

    assert events.frame_event_flags == 0x0000
    assert events.player_hit is False
    assert events.names == ()


def test_events_reply_missing_flag_word_fails_closed():
    with pytest.raises(protocol.HarnessError, match="frameEventFlags"):
        protocol.parse_events_reply("OK frames=7")


def test_events_unavailable_error_is_not_decoded_as_zero():
    with pytest.raises(protocol.HarnessError, match="not OK"):
        protocol.parse_events_reply(REAL_EVENTS_UNAVAILABLE)


def test_events_command_is_sent_once_and_decoded():
    client = _Replay([REAL_EVENTS_PLAYER_HIT_678])

    events = client.events()

    assert client.requests == ["EVENTS"]
    assert events.player_hit is True


# ---------------------------------------------------------------------------
# CHILD command formatting and decoding
# ---------------------------------------------------------------------------


def test_child_hazards_command_and_reply():
    client = _Replay([REAL_HAZARDS_120])

    frame = client.child_hazards(3)

    assert client.requests == ["CHILD 3 HAZARDS"]
    assert frame.frame == 120
    assert (frame.bullet_count, frame.laser_count, frame.enemy_count) == (1, 0, 1)
    assert frame.bullets[0].slot == 1536
    assert frame.enemies[0].world_position[0] == 30.0


def test_child_player_command_and_reply():
    client = _Replay([REAL_PLAYER_120])

    snapshot = client.child_player(4)

    assert client.requests == ["CHILD 4 PLAYER"]
    assert snapshot.speeds_available
    assert snapshot.speed_focused_diagonal == pytest.approx(1.414214, rel=1e-5)


def test_child_events_command_and_reply():
    client = _Replay([REAL_EVENTS_0000])

    events = client.child_events(5)

    assert client.requests == ["CHILD 5 EVENTS"]
    assert events.frame == 120
    assert events.frame_event_flags == 0


def test_child_raw_reply_is_unchanged():
    client = _Replay(["OK advanced=1 frame=121 hash=deadbeefdeadbeef"])

    reply = client.child(6, "STEP 0 1")

    assert client.requests == ["CHILD 6 STEP 0 1"]
    assert reply == "OK advanced=1 frame=121 hash=deadbeefdeadbeef"


def test_child_snapshot_dispatches_by_verb():
    client = _Replay([REAL_HAZARDS_120, REAL_PLAYER_120, REAL_EVENTS_0000, "OK hash=1"])

    assert isinstance(client.child_snapshot(1, "HAZARDS"), protocol.HazardFrame)
    assert isinstance(client.child_snapshot(1, "PLAYER"), protocol.PlayerSnapshot)
    assert isinstance(client.child_snapshot(1, "EVENTS"), protocol.EventFrame)
    assert client.child_snapshot(1, "HASH") == "OK hash=1"


def test_parse_child_reply_unknown_verb_returns_raw_reply():
    assert protocol.parse_child_reply("STATE", "OK frames=1 f0=1") == "OK frames=1 f0=1"


def test_child_commands_reject_malformed_lines():
    client = _Replay([])

    with pytest.raises(ValueError):
        client.child(0, "HASH")
    with pytest.raises(ValueError):
        client.child(1, "HAZARDS\nQUIT")
    with pytest.raises(ValueError):
        client.child(1, "   ")


def test_child_error_reply_is_not_swallowed():
    class _Err(protocol.HarnessClient):
        def __init__(self):
            self.requests = []

        def request(self, line, timeout=None):
            raise protocol.HarnessError(f"{line!r} -> ERR child-command")

    with pytest.raises(protocol.HarnessError, match="child-command"):
        _Err().child_hazards(1)


def test_large_hazards_reply_parses_without_a_line_limit():
    assert len(SYNTHETIC_LARGE_HAZARDS) > 4096

    frame = protocol.parse_child_reply("HAZARDS", SYNTHETIC_LARGE_HAZARDS)

    assert len(frame.bullets) == 64
    assert frame.bullet_count == 64
    assert [row.slot for row in frame.bullets[:3]] == [0, 1, 2]


def test_child_row_decoders_are_the_parent_row_decoders():
    bullets = protocol._rows(
        protocol._parse(REAL_CHILD_BULLETS_120)["bullets"], protocol.parse_bullet_row
    )
    enemies = protocol._rows(
        protocol._parse(REAL_CHILD_ENEMIES_120)["enemies"], protocol.parse_enemy_row
    )
    lasers = protocol._rows(
        protocol._parse(REAL_CHILD_LASERS_120)["lasers"], protocol.parse_laser_row
    )

    assert len(bullets) == len(enemies) == 1
    assert lasers == []
    assert isinstance(bullets[0], protocol.BulletRow)
    assert isinstance(enemies[0], protocol.EnemyRow)


def test_events_are_not_part_of_the_25_field_state():
    client = _Replay([REAL_EVENTS_PLAYER_HIT_678, "OK frames=1 " + " ".join(
        f"f{i}={i}" for i in range(25))])
    client.events()
    state = client.state()

    numbered = {key for key in state if re.fullmatch(r"f\d+", key)}
    assert numbered == {f"f{i}" for i in range(25)}
    assert "frameEventFlags" not in state


# ---------------------------------------------------------------------------
# Live engine: the real fork must answer the read-only snapshots
# ---------------------------------------------------------------------------

BINARY = "/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2/th08-modern"
DATA = "/mnt/d/WorkShop/Touhou_RL/.tmp/calibration"


def _live_ready() -> bool:
    if shutil.which("wsl") is None:
        return False
    probe = subprocess.run(
        [
            "wsl",
            "--exec",
            "sh",
            "-lc",
            f"test -x {BINARY} && test -f {DATA}/th08.dat",
        ],
        capture_output=True,
        timeout=60,
    )
    return probe.returncode == 0


def _run_live(commands: list[str], timeout: float = 240.0) -> list[str]:
    argv = [
        "wsl",
        "--exec",
        "env",
        "-u",
        "DISPLAY",
        "SDL_AUDIODRIVER=dummy",
        "LIBGL_ALWAYS_SOFTWARE=1",
        "GALLIUM_DRIVER=llvmpipe",
        "LP_NUM_THREADS=1",
        BINARY,
        "--data-dir",
        DATA,
        "--rl-control",
        "0,1",
        "--rl-role",
        "shadow",
    ]
    completed = subprocess.run(
        argv,
        input="\n".join(commands) + "\n",
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, (completed.returncode, completed.stdout, completed.stderr)
    lines = completed.stdout.splitlines()
    assert len(lines) == len(commands), (len(lines), len(commands), lines[-3:], completed.stderr)
    return lines


needs_live = pytest.mark.skipif(
    not _live_ready(), reason="build/rl_hazard2 or authentic TH08 calibration data unavailable"
)


@needs_live
def test_live_child_read_only_snapshots_match_the_parent():
    lines = _run_live(
        [
            "HELLO",
            "RESET 0 1 0 1",
            "EVENTS",
            "STEP 0 120",
            "HAZARDS",
            "PLAYER",
            "HASH",
            "EVENTS",
            "PARK",
            "SPAWN 1",
            "CHILD 1 HAZARDS",
            "CHILD 1 PLAYER",
            "CHILD 1 EVENTS",
            "CHILD 1 BULLETS 0 4",
            "CHILD 1 LASERS 0 4",
            "CHILD 1 ENEMIES 0 4",
            "CHILD 1 HASH",
            "CHILD 1 STEP 0 1",
            "CHILD 1 HASH",
            "CHILD 1 QUIT",
            "FRAME",
            "QUIT",
        ]
    )
    (
        hello,
        reset,
        events_after_reset,
        step,
        parent_hazards,
        parent_player,
        parent_hash,
        parent_events,
        park,
        spawn,
        child_hazards,
        child_player,
        child_events,
        child_bullets,
        child_lasers,
        child_enemies,
        child_hash,
        child_step,
        child_hash_after_step,
        child_quit,
        parent_frame_after_child,
        quit_reply,
    ) = lines

    assert "role=shadow" in hello
    assert "ready=1" in reset
    assert events_after_reset.startswith("OK") and "frameEventFlags=" in events_after_reset
    assert "advanced=120" in step
    assert park.startswith("OK park=1 template=1"), park
    assert "spawned=1" in spawn and "handle=1" in spawn, spawn
    assert quit_reply == "OK bye"

    # Same state at fork time -> byte-identical snapshots, not merely "non-empty".
    assert child_hazards == parent_hazards
    assert child_player == parent_player
    assert child_events == parent_events
    assert protocol._parse(child_hash)["hash"] == protocol._parse(parent_hash)["hash"]

    # The child answers the paged read-only commands too.
    for reply, key in (
        (child_bullets, "bullets"),
        (child_lasers, "lasers"),
        (child_enemies, "enemies"),
    ):
        assert reply.startswith("OK"), reply
        assert f"{key}=" in reply
        assert "ERR" not in reply

    # A child step advances the child only; the parent stays at frame 120.
    assert protocol._parse(child_step)["advanced"] == "1"
    assert protocol._parse(child_hash_after_step)["frames"] == "121"
    assert parent_frame_after_child == "OK frame=120"
    assert child_quit == "OK bye"


@needs_live
def test_live_child_forwards_a_hazards_reply_larger_than_4kib():
    lines = _run_live(
        [
            "HELLO",
            "RESET 0 1 0 1",
            "STEP 0 1432",
            "HAZARDS",
            "PARK",
            "SPAWN 1",
            "CHILD 1 HAZARDS",
            "CHILD 1 QUIT",
            "QUIT",
        ],
        timeout=300.0,
    )
    hello, reset, step, parent_hazards, park, spawn, child_hazards, child_quit, quit_reply = lines

    assert "advanced=1432" in step
    assert park.startswith("OK park=1")
    assert spawn.startswith("OK spawned=1")
    # This frame carries 57 bullets and 14 enemies -> far past the 4096-byte
    # line buffer that used to truncate a child reply into ERR child-read.
    assert len(parent_hazards) > 4096, len(parent_hazards)
    assert child_hazards == parent_hazards
    parsed = protocol.parse_hazards_reply(child_hazards)
    assert parsed.bullet_count > 40
    assert parsed.enemy_count >= 10
    assert child_quit == "OK bye"
    assert quit_reply == "OK bye"
