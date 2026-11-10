#!/usr/bin/env python3
"""EXP-P1-004: geometry-vs-engine agreement on real frames (11-P1 §6 WP1).

The P1 teacher spends its budget on the geometric screen and lets the engine
decide.  That split is only sound if the geometric "this frame will collide"
verdict agrees with the engine's own collision verdict on real gameplay frames.
This tool measures that agreement and nothing else.

Two subcommands, because the two halves have different dependencies:

``collect``
    Drives the ``rl_hazard`` harness (``build/rl_hazard/th08-modern``, the build
    that exports ``HAZARDS``/``PLAYER``) through real gameplay with a fixed
    seed and a fixed policy, and writes one JSON line per sampled frame with
    everything needed to re-decide it later: the whole ``HAZARDS`` snapshot,
    ``PLAYER`` (including ``hurtboxBoundsMin/Max``), and the ``STATE`` fields
    that carry the engine's death accounting.  Standard library only, so it
    runs inside the WSL guest where the engine lives.

``analyse``
    Re-reads that JSONL and compares, per model and per frame alignment, the
    11-P1 §2.1 geometry against the engine signal.  Needs numpy, so it runs on
    the Windows side.  It never touches the engine: the dataset is the input.

The engine signal is a **proxy**, and the code says so everywhere it is
computed: TH08 sets ``g_ReplayManager->frameEventFlags |=
REPLAY_FRAME_EVENT_PLAYER_HIT`` (Player.cpp:329/356/466) but the harness exports
none of ``ReplayManager`` (no ``frameEventFlags`` in ``STATE``, ``HASH``,
``DEBUG`` or ``PLAYER``; see ``ENGINE_SIGNALS`` below).  What *is* observable is
the consequence of a hit that was allowed to kill: ``Player::Die()`` sets
``playerState = PLAYER_STATE_DYING`` in the same call that sets the event flag,
and ``CheckLethalCollision`` only reaches ``Die()`` while
``playerState == PLAYER_STATE_ALIVE``.  So

    engine_hit(k) := playerState(k) == ALIVE and playerState(k+1) == DYING

is "one lethal collision committed a death during step k -> k+1".  It is a
strict subset of ``frameEventFlags & 0x2``: the flag also fires while the ship
is dying, spawning or invulnerable, and in those states the flag has no
observable consequence at all.  The evaluation set is therefore restricted to
frames where the ship was ALIVE, which is the only place the proxy and the flag
can be compared.  That restriction is reported, not hidden.

Usage (WSL, engine side)::

    python3 tools/rl/v400_geometry_agreement.py collect \
        --engine /root/Touhou_RL/third_party/th08-rl/build/rl_hazard/th08-modern \
        --data-dir /root/v400-data/geom/game \
        --out .tmp_v400_geom/agreement.jsonl \
        --runs 0:1:still:700 0:1:random:700 3:3:still:500 3:3:random:700 \
        --seed 1

Usage (Windows, analysis side)::

    python tools/rl/v400_geometry_agreement.py analyse \
        --frames .tmp_v400_geom/agreement.jsonl \
        --out .tmp_v400_geom/agreement.report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Buttons, from src/Global.hpp TouhouButton.  BOMB (0x2) and MENU (0x8) are
# never sent: a bomb would make the ship invulnerable (PlayerBomb.cpp:168) and
# a menu press would pause, and both would quietly change what is measured.
BUTTON_SHOOT = 1 << 0
BUTTON_BOMB = 1 << 1
BUTTON_FOCUS = 1 << 2
BUTTON_MENU = 1 << 3
BUTTON_UP = 1 << 4
BUTTON_DOWN = 1 << 5
BUTTON_LEFT = 1 << 6
BUTTON_RIGHT = 1 << 7

#: The 18-action space of 11-P1 §2.2: 9 directions x {unfocused, focused}.
DIRECTION_BITS = (
    0,
    BUTTON_UP,
    BUTTON_DOWN,
    BUTTON_LEFT,
    BUTTON_RIGHT,
    BUTTON_UP | BUTTON_LEFT,
    BUTTON_UP | BUTTON_RIGHT,
    BUTTON_DOWN | BUTTON_LEFT,
    BUTTON_DOWN | BUTTON_RIGHT,
)

#: ``PlayerState`` (Player.hpp:86-92).  ALIVE is 0, DYING is 2.
PLAYER_STATE_ALIVE = 0
PLAYER_STATE_SPAWNING = 1
PLAYER_STATE_DYING = 2
PLAYER_STATE_INVULNERABLE = 3
PLAYER_STATE_NAMES = {
    PLAYER_STATE_ALIVE: "ALIVE",
    PLAYER_STATE_SPAWNING: "SPAWNING",
    PLAYER_STATE_DYING: "DYING",
    PLAYER_STATE_INVULNERABLE: "INVULNERABLE",
}

#: What the engine actually exposes, checked against the harness source.  Kept
#: here (and re-asserted by the analysis) so the experiment card cannot drift
#: away from the code.
ENGINE_SIGNALS = {
    "frameEventFlags": {
        "available": False,
        "why": (
            "ReplayManager::frameEventFlags (ReplayManager.hpp:213, offset 0xda; "
            "REPLAY_FRAME_EVENT_PLAYER_HIT set at Player.cpp:329/356/466) is never "
            "exported: STATE dumps the 25 replay-trace fields only, HASH is the FNV "
            "of those same 25, DEBUG dumps supervisor/gamemanager scalars, PLAYER "
            "dumps player resources and the .sht speed table."
        ),
    },
    "state_field_deaths": {
        "available": True,
        "field": "STATE f6 = FloatBits(globals.deaths)",
        "why": (
            "incremented by GameManager::AddToDeaths at death *commit*, i.e. after "
            "the deathbomb window expires (Player.cpp:1326), so it lags the hit by "
            "2..15 frames and is used only as corroboration."
        ),
    },
    "state_field_lives": {
        "available": True,
        "field": "STATE f7 = FloatBits(globals.livesRemaining)",
        "why": "same commit lag; corroboration only.",
    },
    "state_field_player_state": {
        "available": True,
        "field": "STATE f13 = u8(g_Player.playerState)",
        "why": "alias of PLAYER playerState; kept to cross-check the two reads.",
    },
    "player_state": {
        "available": True,
        "field": "PLAYER playerState",
        "why": (
            "Player::Die() sets PLAYER_STATE_DYING inside the same call that sets "
            "REPLAY_FRAME_EVENT_PLAYER_HIT, and CheckLethalCollision/CheckBulletCollision "
            "reach Die() only while playerState == ALIVE.  The ALIVE->DYING edge is "
            "therefore an observable, one-frame-accurate stand-in for a hit that was "
            "allowed to kill.  PROXY, not the event flag itself."
        ),
    },
    "player_invuln_state": {
        "available": True,
        "field": "PLAYER invulnState",
        "why": "1 iff playerState == INVULNERABLE, so it adds nothing beyond playerState.",
    },
    "hurtbox_bounds": {
        "available": True,
        "field": "PLAYER hurtboxMin{X,Y} / hurtboxMax{X,Y}",
        "why": "the exact box the engine tests; geometry uses it instead of position +/- half.",
    },
}

#: Reply fields of a ``HazardFrame`` we keep.  Chosen so a frame is replayable
#: without the engine but still cheap.  Bullets keep their velocity because the
#: engine tests a bullet *after* ``position += velocity`` (BulletManager.cpp:
#: 862-863) and despawns the one that hit inside that same block, so the killing
#: bullet is invisible at both frames it spans unless the velocity can put it
#: back (see ``geometry.MODEL_ENGINE_PHASE``).
BULLET_FIELDS = ("slot", "state", "collision_size", "collision_disabled",
                 "x", "y", "vx", "vy", "active_timer")
ENEMY_FIELDS = ("slot", "flags1", "wx", "wy", "hitbox_width", "hitbox_height")
LASER_FIELDS = ("slot", "state", "in_use", "x", "y", "angle", "start", "end", "width")


# ---------------------------------------------------------------------------
# collect: engine side (standard library only)
# ---------------------------------------------------------------------------
class EngineError(RuntimeError):
    """The engine replied ``ERR`` or the control pipe died.

    ``recoverable`` separates the two cases that matter to the caller: an
    ``ERR`` reply means the process is alive and a fresh ``RESET`` can put it
    back into gameplay (a game over ends a run, not the session), while a dead
    pipe means no further real frames can be collected at all.
    """

    def __init__(self, message: str, *, recoverable: bool = False) -> None:
        super().__init__(message)
        self.recoverable = recoverable


class ControlClient:
    """Bare line-protocol client, so the collector needs no repo imports."""

    def __init__(self, binary: str, data_dir: str, *, role: str = "shadow",
                 timeout: float = 300.0, stderr_path: Path | None = None) -> None:
        self.binary = str(binary)
        self.data_dir = str(data_dir)
        self.role = role
        self.timeout = float(timeout)
        self.stderr_path = stderr_path
        self._proc: subprocess.Popen[bytes] | None = None
        self._buffer = b""
        self._stderr = None
        self.requests = 0

    def start(self) -> None:
        env = dict(os.environ)
        for name in ("DISPLAY", "WAYLAND_DISPLAY"):
            env.pop(name, None)
        env.update(
            SDL_AUDIODRIVER="dummy",
            LIBGL_ALWAYS_SOFTWARE="1",
            GALLIUM_DRIVER="llvmpipe",
            LP_NUM_THREADS="1",
        )
        if self.stderr_path is not None:
            self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr = open(self.stderr_path, "wb")
        argv = [self.binary, "--data-dir", self.data_dir,
                "--rl-control", "0,1", "--rl-role", self.role]
        self._proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr or subprocess.DEVNULL, env=env, bufsize=0,
        )

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    self._write("QUIT")
                    self._readline(time.monotonic() + 20.0)
                except Exception:
                    pass
            try:
                proc.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        finally:
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
            if self._stderr is not None:
                try:
                    self._stderr.close()
                except Exception:
                    pass
                self._stderr = None
            self._proc = None
            self._buffer = b""

    def _write(self, line: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((line + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def _readline(self, deadline: float) -> str:
        import select

        assert self._proc is not None and self._proc.stdout is not None
        while time.monotonic() < deadline:
            if b"\n" in self._buffer:
                line, _, self._buffer = self._buffer.partition(b"\n")
                return line.decode("utf-8", "replace")
            remaining = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self._proc.stdout], [], [], min(0.5, remaining))
            if not ready:
                if self._proc.poll() is not None:
                    raise EngineError("engine exited before replying")
                continue
            chunk = os.read(self._proc.stdout.fileno(), 65536)
            if not chunk:
                raise EngineError("engine closed the control pipe")
            self._buffer += chunk
        raise EngineError("no reply within the timeout")

    def request(self, line: str, timeout: float | None = None) -> str:
        if self._proc is None:
            raise EngineError("client is not running")
        self._write(line)
        reply = self._readline(time.monotonic() + (self.timeout if timeout is None else timeout))
        self.requests += 1
        if reply.startswith("ERR"):
            raise EngineError(f"{line!r} -> {reply}", recoverable=True)
        return reply


def parse_reply(reply: str) -> dict[str, str]:
    parts = reply.split()
    out = {"_verb": parts[0] if parts else ""}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def bits_to_float(token: str) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<I", int(token, 16) & 0xFFFFFFFF))[0]


def decode_player(reply: str) -> dict:
    fields = parse_reply(reply)
    return {
        "frame": int(fields.get("frames", 0)),
        "hp": bits_to_float(fields["hp"]),
        "lives": int(fields["lives"]),
        "bombs": bits_to_float(fields.get("bombs", "0")),
        "power": bits_to_float(fields.get("power", "0")),
        "focus": int(fields.get("focus", 0)),
        "youkai": int(fields.get("isYoukai", 0)),
        "state": int(fields.get("playerState", 0)),
        "invuln": int(fields.get("invulnState", 0)),
        "pos": [bits_to_float(fields["px"]), bits_to_float(fields["py"])],
        "hmin": [bits_to_float(fields["hurtboxMinX"]), bits_to_float(fields["hurtboxMinY"])],
        "hmax": [bits_to_float(fields["hurtboxMaxX"]), bits_to_float(fields["hurtboxMaxY"])],
        "speeds": [
            bits_to_float(fields["speedNormalAxial"]),
            bits_to_float(fields["speedNormalDiagonal"]),
            bits_to_float(fields["speedFocusedAxial"]),
            bits_to_float(fields["speedFocusedDiagonal"]),
        ],
    }


def decode_state(reply: str) -> dict:
    fields = parse_reply(reply)
    return {
        "frame": int(fields.get("frames", 0)),
        "deaths": bits_to_float(fields["f6"]),
        "lives_remaining": bits_to_float(fields["f7"]),
        "player_state": int(fields["f13"]),
        "graze": int(fields["f5"]),
    }


def decode_hazards(reply: str) -> dict:
    """Decode one ``HAZARDS`` reply into compact per-family arrays."""

    fields = parse_reply(reply)
    if fields.get("_verb") != "OK":
        raise EngineError(f"HAZARDS reply is not OK: {reply[:120]!r}")

    bullets = []
    for row in fields.get("bullets", "").split(","):
        if not row:
            continue
        parts = row.split(":")
        if len(parts) < 11:
            raise EngineError(f"bullet row has {len(parts)} fields, needs 11: {row!r}")
        bullets.append([
            int(parts[0]), int(parts[1]),
            bits_to_float(parts[9]), int(parts[10]),
            bits_to_float(parts[3]), bits_to_float(parts[4]),
            bits_to_float(parts[6]), bits_to_float(parts[7]),
            int(parts[8]),
        ])

    enemies = []
    for row in fields.get("enemies", "").split(","):
        if not row:
            continue
        parts = row.split(":")
        if len(parts) < 12:
            raise EngineError(f"enemy row has {len(parts)} fields, needs 12: {row!r}")
        enemies.append([
            int(parts[0]), int(parts[1], 16),
            bits_to_float(parts[7]), bits_to_float(parts[8]),
            bits_to_float(parts[10]), bits_to_float(parts[11]),
        ])

    lasers = []
    for row in fields.get("lasers", "").split(","):
        if not row:
            continue
        parts = row.split(":")
        if len(parts) < 12:
            raise EngineError(f"laser row has {len(parts)} fields, needs 12: {row!r}")
        lasers.append([
            int(parts[0]), int(parts[1]), int(parts[11]),
            bits_to_float(parts[2]), bits_to_float(parts[3]),
            bits_to_float(parts[5]),
            bits_to_float(parts[6]), bits_to_float(parts[7]),
            bits_to_float(parts[9]),
        ])

    return {
        "frame": int(fields.get("frames", 0)),
        "nb": int(fields["nb"]), "nl": int(fields["nl"]), "ne": int(fields["ne"]),
        "b": bullets, "e": enemies, "l": lasers,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Policy:
    """Seeded action source for both policies.

    ``still`` holds the ship in place (mask 0 is also valid, but holding SHOOT
    matches how a real player always fires and keeps the two policies differing
    only in movement).  ``random`` resamples one of the 18 directions of
    11-P1 §2.2 every ``hold`` frames from a seeded RNG, so a run is reproducible
    from ``policy_seed`` alone and the ship actually travels instead of
    jittering in place.
    """

    def __init__(self, name: str, rng: random.Random, hold: int = 6) -> None:
        if name not in ("still", "random"):
            raise SystemExit(f"unknown policy {name!r} (use still/random)")
        self.name = name
        self.rng = rng
        self.hold = max(1, int(hold))
        self._action = self._sample()

    def _sample(self) -> int:
        direction = self.rng.choice(DIRECTION_BITS)
        focus = BUTTON_FOCUS if self.rng.random() < 0.35 else 0
        return BUTTON_SHOOT | direction | focus

    def mask(self, index: int) -> int:
        if self.name == "still":
            return BUTTON_SHOOT
        if index % self.hold == 0:
            self._action = self._sample()
        return self._action


def collect(args) -> int:
    engine = Path(args.engine)
    if not engine.is_file():
        raise SystemExit(f"engine binary not found: {engine}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Truncate: a leftover dataset from a previous invocation would otherwise be
    # spliced into this one and silently change the sample size.
    out.write_bytes(b"")
    stderr_path = out.with_suffix(".engine.stderr.log")

    meta = {
        "engine": str(engine),
        "engine_sha256": sha256_file(engine),
        "data_dir": str(Path(args.data_dir).resolve()),
        "role": args.role,
        "seed": args.seed,
        "team": 0,
        "runs": args.runs,
        "engine_signals": ENGINE_SIGNALS,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    frames_written = 0
    runs_meta = []
    requests = 0
    started = time.monotonic()
    # One engine process per run.  A second ``RESET`` on a process that has
    # already played a stage comes back ``ready=0`` and ``HAZARDS`` then replies
    # ``ERR hazards-not-ready reset-required``: the setup subthread never
    # finishes a re-entry, so a session cannot be reused.  Respawning is also
    # what keeps a game over in one run from costing the rest of the dataset.
    for run_index, spec in enumerate(args.runs):
        bits = spec.split(":")
        if len(bits) not in (4, 5, 6):
            raise SystemExit(
                f"bad --runs entry {spec!r} "
                "(stage:difficulty:policy:frames[:policySeed[:resetSeed]])"
            )
        stage, difficulty, policy, frames = (
            int(bits[0]), int(bits[1]), bits[2], int(bits[3])
        )
        policy_seed = int(bits[4]) if len(bits) >= 5 else args.seed * 1000 + run_index
        # A distinct RESET seed is the only thing that makes two runs of the
        # same stage differ: the harness is deterministic, so repeating
        # (stage, difficulty, seed, policy) would replay the very same frames.
        reset_seed = int(bits[5]) if len(bits) == 6 else args.seed
        actions = Policy(policy, random.Random(policy_seed), args.hold)
        run_started = time.monotonic()
        client = ControlClient(
            engine, args.data_dir, role=args.role, timeout=args.timeout,
            stderr_path=stderr_path.with_name(
                f"{stderr_path.stem}.run{run_index}{stderr_path.suffix}"
            ),
        )
        samples = []
        recorded = 0
        error = None
        reset = None
        hello = None
        try:
            client.start()
            hello = client.request("HELLO")
            reset = client.request(
                f"RESET {stage} {difficulty} 0 {reset_seed} 0", timeout=240.0
            )
            if "ready=0" in reset:
                raise EngineError(f"RESET did not become ready: {reset}", recoverable=False)
            # Frames 0..warmup of a stage are the banner/spawn window, where the
            # ship is INVULNERABLE and cannot die; stepping through them keeps
            # stage time real while spending no sample budget on frames the
            # comparison has to exclude anyway.
            for index in range(-args.warmup, frames):
                mask = actions.mask(index if index >= 0 else 0)
                client.request(f"STEP {mask} 1")
                if index < 0:
                    continue
                hazards = decode_hazards(client.request("HAZARDS"))
                player = decode_player(client.request("PLAYER"))
                state = decode_state(client.request("STATE"))
                samples.append({
                    "run": run_index,
                    "i": index,
                    "stage": stage,
                    "difficulty": difficulty,
                    "policy": policy,
                    "mask": mask,
                    "frame": hazards["frame"],
                    "player": player,
                    "state": state,
                    "nb": hazards["nb"], "nl": hazards["nl"], "ne": hazards["ne"],
                    "b": hazards["b"], "e": hazards["e"], "l": hazards["l"],
                })
                recorded += 1
                if args.flush_every and len(samples) >= args.flush_every:
                    _append(out, samples)
                    frames_written += len(samples)
                    samples = []
        except EngineError as exc:
            # An ERR reply (typically a game over) ends this run but not the
            # session, because the next run gets a fresh process anyway.
            error = str(exc)
        finally:
            client.close()
            requests += client.requests
        _append(out, samples)
        frames_written += len(samples)
        runs_meta.append({
            "run": run_index, "stage": stage, "difficulty": difficulty,
            "policy": policy, "frames_requested": frames,
            "frames_recorded": recorded, "policy_seed": policy_seed,
            "reset_reply": reset, "hello": hello,
            "error": error,
            "wall_seconds": round(time.monotonic() - run_started, 3),
        })

    meta["runs_meta"] = runs_meta
    meta["frames_written"] = frames_written
    meta["requests"] = requests
    meta["wall_seconds"] = round(time.monotonic() - started, 3)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "frames_written": frames_written,
        "runs": len(runs_meta),
        "wall_seconds": meta["wall_seconds"],
        "engine_sha256": meta["engine_sha256"],
        "out": str(out),
    }, indent=2))
    return 0


def _append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# analyse: numpy side, dataset in, verdict out
# ---------------------------------------------------------------------------
def load_frames(path: Path) -> list[dict]:
    frames = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def _rehydrate(frame: dict):
    """Rebuild the protocol dataclasses so geometry sees the real types."""

    from touhou_rl.v400.gym.protocol import (
        BulletRow,
        EnemyRow,
        HazardFrame,
        LaserRow,
        PlayerSnapshot,
    )

    bullets = tuple(
        BulletRow(
            slot=row[0], state=row[1], color=0,
            position=(row[4], row[5], 0.0), velocity=(row[6], row[7]),
            timer=row[8],
            collision_size=row[2], collision_disabled=row[3],
        )
        for row in frame["b"]
    )
    enemies = tuple(
        EnemyRow(
            slot=row[0], flags1=row[1], position=(0.0, 0.0, 0.0), life=0, timer=0,
            world_position=(row[2], row[3], 0.0),
            hitbox_width=row[4], hitbox_height=row[5],
        )
        for row in frame["e"]
    )
    lasers = tuple(
        LaserRow(
            slot=row[0], state=row[1], position=(row[3], row[4], 0.0),
            angle=row[5], start_offset=row[6], end_offset=row[7],
            current_width=row[7] - row[6], width=row[8], timer=0, in_use=row[2],
        )
        for row in frame["l"]
    )
    hazards = HazardFrame(
        frame=frame["frame"], bullet_count=frame["nb"], laser_count=frame["nl"],
        enemy_count=frame["ne"], bullets=bullets, lasers=lasers, enemies=enemies,
    )
    player = PlayerSnapshot(
        frame=frame["player"]["frame"],
        hp=frame["player"]["hp"], lives=frame["player"]["lives"],
        bombs=frame["player"]["bombs"], power=frame["player"]["power"],
        focus=frame["player"]["focus"], is_youkai=frame["player"]["youkai"],
        player_state=frame["player"]["state"], invuln_state=frame["player"]["invuln"],
        position=tuple(frame["player"]["pos"]),
        hurtbox_min=tuple(frame["player"]["hmin"]),
        hurtbox_max=tuple(frame["player"]["hmax"]),
        speed_normal_axial=frame["player"]["speeds"][0],
        speed_normal_diagonal=frame["player"]["speeds"][1],
        speed_focused_axial=frame["player"]["speeds"][2],
        speed_focused_diagonal=frame["player"]["speeds"][3],
    )
    return hazards, player


def _confusion(rows: list[tuple[int, int]]) -> dict:
    """rows are ``(geom, engine)``; returns the 2x2 plus the headline rate."""

    tp = sum(1 for geom, eng in rows if geom and eng)
    fp = sum(1 for geom, eng in rows if geom and not eng)
    fn = sum(1 for geom, eng in rows if not geom and eng)
    tn = sum(1 for geom, eng in rows if not geom and not eng)
    total = len(rows)
    agreed = tp + tn
    return {
        "n": total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "agreed": agreed,
        "agreement": (agreed / total) if total else None,
        "disagreements": fp + fn,
    }


def bullet_motion_probe(frames: list[dict], next_index: list[int | None]) -> dict:
    """Measure how far a bullet actually moves per frame, against its velocity.

    The engine-phase model rests on ``position_{k+1} == position_k + velocity_k``
    (BulletManager.cpp:862-863).  If that identity holds for the bullets that
    survive into the next snapshot, the one-tick advance is the engine's own
    arithmetic and not an approximation.  Bullets under an ECL transform move
    differently, so the report keeps the error distribution instead of a single
    number.
    """

    matched = 0
    worst = 0.0
    errors: list[float] = []
    for index, frame in enumerate(frames):
        following = next_index[index]
        if following is None:
            continue
        before = {
            row[0]: row for row in frame["b"] if row[1] == 1 and not row[3]
        }
        after = {
            row[0]: row for row in frames[following]["b"] if row[1] == 1 and not row[3]
        }
        for slot in before.keys() & after.keys():
            row = before[slot]
            measured_x = after[slot][4] - row[4]
            measured_y = after[slot][5] - row[5]
            error = max(abs(measured_x - row[6]), abs(measured_y - row[7]))
            errors.append(error)
            matched += 1
            worst = max(worst, error)
    errors.sort()
    return {
        "matched_bullet_pairs": matched,
        "max_abs_error_px": worst,
        "p50_abs_error_px": errors[len(errors) // 2] if errors else None,
        "p99_abs_error_px": errors[int(len(errors) * 0.99)] if errors else None,
        "pairs_within_1e-4": sum(1 for value in errors if value <= 1e-4),
    }


def load_geometry():
    """Import ``touhou_rl.v400.teacher.geometry`` without running its ``__init__``.

    The teacher package is being built by a parallel work package, and while
    that is in flight its ``__init__`` re-exports modules that may not import
    yet.  Loading the module straight from its file keeps this measurement
    reproducible against exactly the file under test, instead of depending on
    whether the sibling package's re-export list is currently green.
    """

    import importlib.util
    import sys as _sys

    path = REPO_ROOT / "touhou_rl" / "v400" / "teacher" / "geometry.py"
    if not path.is_file():
        raise SystemExit(f"geometry module not found: {path}")
    module_name = "_v400_teacher_geometry"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__], so
    # the module has to be registered before its body runs.
    _sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def analyse(args) -> int:
    geometry = load_geometry()
    frames_path = Path(args.frames)
    frames = load_frames(frames_path)
    if not frames:
        raise SystemExit(f"no frames in {frames_path}")

    meta_path = frames_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    # --- sample bookkeeping -------------------------------------------------
    by_run_index: dict[int, list[int]] = {}
    for index, frame in enumerate(frames):
        by_run_index.setdefault(frame["run"], []).append(index)

    continuity = {}
    for run, indices in by_run_index.items():
        rows = [frames[index] for index in indices]
        numbers = [row["frame"] for row in rows]
        gaps = [b - a for a, b in zip(numbers, numbers[1:])]
        continuity[run] = {
            "frames": len(rows),
            "stage": rows[0]["stage"],
            "difficulty": rows[0]["difficulty"],
            "policy": rows[0]["policy"],
            "first_frame": numbers[0],
            "last_frame": numbers[-1],
            "max_gap": max(gaps) if gaps else 0,
            "gaps_not_one": sum(1 for gap in gaps if gap != 1),
            "player_states": _histogram([row["player"]["state"] for row in rows]),
            "max_bullets": max(row["nb"] for row in rows),
            "max_lasers": max(row["nl"] for row in rows),
            "max_enemies": max(row["ne"] for row in rows),
            "laser_frames": sum(1 for row in rows if row["nl"] > 0),
        }

    # --- next-frame links and engine events --------------------------------
    # ev[k] needs frame k+1 of the SAME run; a run boundary is never bridged.
    engine_hit = [0] * len(frames)
    next_state: list[int | None] = [None] * len(frames)
    next_index: list[int | None] = [None] * len(frames)
    for run, indices in by_run_index.items():
        for position, index in enumerate(indices[:-1]):
            following_index = indices[position + 1]
            next_index[index] = following_index
            next_state[index] = frames[following_index]["player"]["state"]
            if frames[index]["player"]["state"] == PLAYER_STATE_ALIVE and \
                    frames[following_index]["player"]["state"] == PLAYER_STATE_DYING:
                engine_hit[index] = 1

    deaths_rise = [0] * len(frames)
    lives_fall = [0] * len(frames)
    for index in range(len(frames)):
        following_index = next_index[index]
        if following_index is None:
            continue
        if frames[following_index]["state"]["deaths"] > frames[index]["state"]["deaths"]:
            deaths_rise[index] = 1
        if frames[following_index]["state"]["lives_remaining"] < \
                frames[index]["state"]["lives_remaining"]:
            lives_fall[index] = 1

    alive = [frame["player"]["state"] == PLAYER_STATE_ALIVE for frame in frames]
    usable = [next_index[index] is not None for index in range(len(frames))]
    evaluated = [alive[index] and usable[index] for index in range(len(frames))]

    # --- geometric verdict per frame, per model ----------------------------
    snapshots = [_rehydrate(frame) for frame in frames]
    verdicts: dict[str, list] = {}
    for model_name, model in geometry.MODELS.items():
        phase = geometry.HURTBOX_PHASE.get(model_name, "current")
        out = []
        for index in range(len(frames)):
            hazards, player = snapshots[index]
            if phase == "next":
                following = next_index[index]
                if following is None:
                    out.append(None)
                    continue
                player = snapshots[following][1]
            out.append(geometry.frame_verdict(hazards, player, model))
        verdicts[model_name] = out

    report: dict = {
        "dataset": {
            "path": str(frames_path),
            "sha256": sha256_file(frames_path),
            "frames": len(frames),
            "frames_alive": sum(1 for flag in alive if flag),
            "frames_usable": sum(1 for flag in usable if flag),
            "frames_alive_usable": sum(evaluated),
            "engine_hits_on_evaluated": sum(
                engine_hit[index] for index in range(len(frames)) if evaluated[index]
            ),
            "engine_hits_total": sum(engine_hit),
            "death_commit_events": sum(deaths_rise),
            "lives_drops": sum(lives_fall),
            "runs": continuity,
            "engine": meta.get("engine"),
            "engine_sha256": meta.get("engine_sha256"),
            "role": meta.get("role"),
            "seed": meta.get("seed"),
            "engine_signals": meta.get("engine_signals", ENGINE_SIGNALS),
            "bullet_motion_probe": bullet_motion_probe(frames, next_index),
        },
        "models": {},
    }

    for model_name in verdicts:
        pred = [None if verdict is None else verdict.hit for verdict in verdicts[model_name]]
        rows = [
            (int(pred[index]), int(engine_hit[index]))
            for index in range(len(frames))
            if evaluated[index]
        ]
        all_rows = [
            (int(pred[index]), int(engine_hit[index]))
            for index in range(len(frames))
            if usable[index] and pred[index] is not None
        ]
        report["models"][model_name] = {
            "hurtbox_phase": geometry.HURTBOX_PHASE.get(model_name, "current"),
            "bullet_advance": geometry.MODELS[model_name].bullet_advance,
            "confusion_alive_forward": _confusion(rows),
            "confusion_all_frames": _confusion(all_rows),
        }

        detail = []
        for index in range(len(frames)):
            if not evaluated[index]:
                continue
            if pred[index] is None or int(pred[index]) == int(engine_hit[index]):
                continue
            frame = frames[index]
            verdict = verdicts[model_name][index]
            following = next_index[index]
            detail.append({
                "run": frame["run"],
                "i": frame["i"],
                "frame": frame["frame"],
                "geom": int(pred[index]),
                "engine": int(engine_hit[index]),
                "player_state": frame["player"]["state"],
                "player_state_next": next_state[index],
                "nearest": verdict.as_dict()["nearest"],
                "hit_kinds": verdict.as_dict()["hit_kinds"],
                "bullets": frame["nb"],
                "enemies": frame["ne"],
                "lasers": frame["nl"],
                "hurtbox": [frame["player"]["hmin"], frame["player"]["hmax"]],
                "player_pos": frame["player"]["pos"],
                "damageable_enemies": sum(
                    1 for row in frame["e"]
                    if row[1] & geometry.ENEMY_FLAG_COLLISION
                    and not row[1] & geometry.ENEMY_FLAG_SUPPRESSING
                ),
                "collision_enemies": sum(
                    1 for row in frame["e"] if row[1] & geometry.ENEMY_FLAG_COLLISION
                ),
                "geom_at_next": (
                    None if following is None or pred[following] is None
                    else int(pred[following])
                ),
                "despawned_bullets_next": (
                    0 if following is None
                    else sum(
                        1 for row in frames[following]["b"]
                        if row[1] != 1 and row[0] in {r[0] for r in frame["b"]}
                    )
                ),
            })
        report["models"][model_name]["mismatches"] = detail
        report["models"][model_name]["mismatch_count"] = len(detail)
        report["models"][model_name]["mismatch_classes"] = _histogram([
            f"geom{row['geom']}_engine{row['engine']}" for row in detail
        ])

    # --- precision control --------------------------------------------------
    f64_model = geometry.Model(
        name="spec-float64", bullet_half_scale=0.5, enemy_half_divisor=3.0,
        inclusive=True, dtype="float64",
    )
    f64_hits = [
        geometry.frame_verdict(snapshots[index][0], snapshots[index][1], f64_model).hit
        for index in range(len(frames))
    ]
    f32_hits = [verdicts["spec"][index].hit for index in range(len(frames))]
    report["precision_control"] = {
        "frames": len(frames),
        "frames_where_float64_disagrees": sum(
            1 for a, b in zip(f32_hits, f64_hits) if bool(a) != bool(b)
        ),
        "confusion_float64": _confusion([
            (int(f64_hits[index]), int(engine_hit[index]))
            for index in range(len(frames)) if evaluated[index]
        ]),
    }

    # --- verdict ------------------------------------------------------------
    headline = report["models"]["engine-phase"]["confusion_alive_forward"]
    threshold = 0.999
    agreement = headline["agreement"] if headline["agreement"] is not None else 0.0
    report["verdict"] = {
        "threshold": threshold,
        "headline_model": "engine-phase",
        "agreement": agreement,
        "passes": bool(agreement >= threshold),
        "usable_for_beam_search": bool(agreement >= threshold and headline["n"] >= 1000),
        "n": headline["n"],
        "spec_agreement": report["models"]["spec"]["confusion_alive_forward"]["agreement"],
        "note": (
            "agreement is measured against the ALIVE->DYING proxy on frames where "
            "the ship was alive, the only frames on which PLAYER_HIT can be "
            "observed as a death; see ENGINE_SIGNALS.player_state"
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "frames": len(frames),
        "frames_alive_usable": report["dataset"]["frames_alive_usable"],
        "engine_hits": report["dataset"]["engine_hits_on_evaluated"],
        "bullet_motion_probe": report["dataset"]["bullet_motion_probe"],
        "per_model": {
            name: report["models"][name]["confusion_alive_forward"] for name in report["models"]
        },
        "mismatch_classes": {
            name: report["models"][name]["mismatch_classes"] for name in report["models"]
        },
        "precision_disagreements": report["precision_control"]["frames_where_float64_disagrees"],
        "verdict": report["verdict"],
        "out": str(out),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _histogram(values: list[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    collect_parser = sub.add_parser("collect", help="drive the engine and record frames")
    collect_parser.add_argument("--engine", required=True)
    collect_parser.add_argument("--data-dir", required=True)
    collect_parser.add_argument("--out", required=True)
    collect_parser.add_argument("--role", default="shadow")
    collect_parser.add_argument("--seed", type=int, default=1)
    collect_parser.add_argument("--hold", type=int, default=6,
                                help="frames a random-policy direction is held")
    collect_parser.add_argument("--warmup", type=int, default=150,
                                help="frames stepped but not recorded, to skip the "
                                     "stage-start invulnerability window")
    collect_parser.add_argument("--flush-every", type=int, default=200)
    collect_parser.add_argument("--timeout", type=float, default=300.0)
    collect_parser.add_argument(
        "--runs", nargs="+", required=True,
        help="stage:difficulty:policy:frames[:policySeed] ... (still|random)",
    )
    collect_parser.set_defaults(func=collect)

    analyse_parser = sub.add_parser("analyse", help="compare recorded frames")
    analyse_parser.add_argument("--frames", required=True)
    analyse_parser.add_argument("--out", required=True)
    analyse_parser.set_defaults(func=analyse)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
