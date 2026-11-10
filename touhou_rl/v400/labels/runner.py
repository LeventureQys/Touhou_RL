"""Engine-side collector for the P2 v2 label dataset (EXP-P2-002).

One *shard* is one bounded engine run: ``RESET`` (or replay playback), a warm-up
to the first ``ALIVE`` frame, then ``frames`` recorded frames.  Every recorded
frame is written as one JSONL label line (state-derived labels v2, real
``EVENTS`` bits) plus its own pixels.  The label for frame ``k`` is written only
after frame ``k+16`` exists, because :func:`evaluate_future_danger` needs the
recorded future hazard frames; the tail of a run is emitted with fewer
successors and the label says so instead of guessing.

Channel discipline (fail-closed, no invented values):

* ``HAZARDS`` needs ``g_resetReady`` (``rlharness.cpp`` ``HandleHazards``), so it
  is unavailable during replay playback.  Replay shards therefore carry
  ``occupancy=None`` and ``future_danger=None`` with an explicit reason, and
  still carry ``STATE``/``PLAYER``/``EVENTS``/pixels, all of which work there.
* ``PLAYER`` needs only ``g_GameManager.globals``; before the stage starts it
  replies ``ERR`` and the frame's player label is all-``None`` with
  ``missing`` listing every field.
* ``EVENTS`` needs ``g_ReplayManager``; if it ever replies ``ERR`` the shard
  records ``events_error`` and *every* frame's EVENTS bits are ``None`` — never
  a fabricated ``0``.

The module is import-safe on Windows (no numpy / no engine at import time); the
engine-facing paths are only exercised inside the WSL guest.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Buttons, from ``src/Global.hpp`` ``TouhouButton``.
BUTTON_SHOOT = 1 << 0
BUTTON_BOMB = 1 << 1
BUTTON_FOCUS = 1 << 2
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

#: Sources this collector can really drive.
SOURCES = ("random", "still", "replay")

#: How many future frames the delayed writer holds before emitting a label.
FUTURE_HORIZON = 16


class EngineError(RuntimeError):
    """The engine replied ``ERR`` or the control pipe died."""


class SourceUnavailable(RuntimeError):
    """A requested source cannot be driven by this collector (fail closed)."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_reply(reply: str) -> dict[str, str]:
    """``OK a=1 b=2`` -> ``{"_verb": "OK", "a": "1", "b": "2"}``."""

    parts = reply.split()
    out: dict[str, str] = {"_verb": parts[0] if parts else ""}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


class ControlClient:
    """Bare line-protocol client for the ``env`` role (keeps DISPLAY for GL)."""

    def __init__(
        self,
        binary: str | Path,
        data_dir: str | Path,
        *,
        role: str = "env",
        timeout: float = 300.0,
        stderr_path: Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.binary = str(binary)
        self.data_dir = str(data_dir)
        self.role = role
        self.timeout = float(timeout)
        self.stderr_path = stderr_path
        self.extra_env = dict(extra_env or {})
        self._proc: subprocess.Popen[bytes] | None = None
        self._buffer = b""
        self._stderr = None
        self.requests = 0

    def start(self) -> None:
        env = dict(os.environ)
        env.update(
            SDL_AUDIODRIVER="dummy",
            LIBGL_ALWAYS_SOFTWARE="1",
            GALLIUM_DRIVER="llvmpipe",
            LP_NUM_THREADS="1",
        )
        env.update(self.extra_env)
        if self.stderr_path is not None:
            self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr = open(self.stderr_path, "wb")
        argv = [
            self.binary,
            "--data-dir",
            self.data_dir,
            "--rl-control",
            "0,1",
            "--rl-role",
            self.role,
        ]
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr or subprocess.DEVNULL,
            env=env,
            bufsize=0,
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
        reply = self._readline(
            time.monotonic() + (self.timeout if timeout is None else timeout)
        )
        self.requests += 1
        if reply.startswith("ERR"):
            raise EngineError(f"{line!r} -> {reply}")
        return reply

    def try_request(self, line: str, timeout: float | None = None) -> tuple[str | None, str | None]:
        """``(reply, None)`` on success, ``(None, error)`` on ``ERR``/timeout.

        Used for channels that are legitimately unavailable in some modes, so an
        unavailable channel becomes a recorded reason instead of a crashed run.
        """

        try:
            return self.request(line, timeout=timeout), None
        except EngineError as exc:
            return None, str(exc)


class Policy:
    """Seeded action source: ``still`` (shoot only) or the 18-action random one.

    ``random`` follows 12-P2 §2.1 — direction and focus resampled every ``hold``
    frames, shoot always held, and an *occasional* bomb (``bomb_prob`` per
    decision) so that real ``BOMB_STARTED`` frames exist.  The bomb probability
    is a collection parameter, recorded in the shard manifest; it is not a
    property of the engine.
    """

    def __init__(
        self,
        name: str,
        rng: random.Random,
        hold: int = 6,
        bomb_prob: float = 0.0,
    ) -> None:
        if name not in ("still", "random"):
            raise SourceUnavailable(f"policy {name!r} is not an engine policy")
        self.name = name
        self.rng = rng
        self.hold = max(1, int(hold))
        self.bomb_prob = float(bomb_prob)
        self.bombs_pressed = 0
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
        action = self._action
        if self.bomb_prob > 0.0 and index % self.hold == 0:
            if self.rng.random() < self.bomb_prob:
                action |= BUTTON_BOMB
                self.bombs_pressed += 1
        return action


@dataclass
class ShardSpec:
    """One bounded collection run; its ``key`` is the resume identity."""

    source: str
    stage: int
    difficulty: int
    seed: int
    frames: int
    role: str = "env"
    hold: int = 6
    bomb_prob: float = 0.0
    warmup: int = 700
    timeout: float = 300.0
    replay_path: str | None = None
    pixel_format: str = "png"
    tag: str = ""

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise SourceUnavailable(
                f"source {self.source!r} not supported by this collector: {SOURCES}"
            )
        if self.source == "replay" and not self.replay_path:
            raise SourceUnavailable("source='replay' needs --replay-path")

    @property
    def key(self) -> str:
        base = f"{self.source}-s{self.stage}d{self.difficulty}-seed{self.seed}-n{self.frames}"
        if self.tag:
            base += f"-{self.tag}"
        if self.source == "random" and self.bomb_prob > 0.0:
            base += f"-bp{self.bomb_prob:g}"
        return base

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "stage": self.stage,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "frames": self.frames,
            "role": self.role,
            "hold": self.hold,
            "bomb_prob": self.bomb_prob,
            "warmup": self.warmup,
            "replay_path": self.replay_path,
            "pixel_format": self.pixel_format,
            "tag": self.tag,
            "key": self.key,
        }


def next_attempt_dir(shard_root: Path) -> Path:
    """``shards/<key>/attempt-NN`` with the next free NN (never overwrites)."""

    existing = sorted(
        int(path.name.split("-", 1)[1])
        for path in shard_root.glob("attempt-*")
        if path.is_dir() and path.name.split("-", 1)[1].isdigit()
    )
    number = (existing[-1] + 1) if existing else 1
    return shard_root / f"attempt-{number:02d}"


def latest_attempt(shard_root: Path) -> Path | None:
    attempts = sorted(
        (path for path in shard_root.glob("attempt-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    return attempts[-1] if attempts else None


def attempt_meta_path(attempt_dir: Path) -> Path:
    return attempt_dir / "shard.json"


def read_attempt_meta(attempt_dir: Path) -> dict[str, Any] | None:
    path = attempt_meta_path(attempt_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def shard_complete(shard_root: Path) -> dict[str, Any] | None:
    """The latest attempt's meta when it is complete, else ``None``."""

    attempt = latest_attempt(shard_root)
    if attempt is None:
        return None
    meta = read_attempt_meta(attempt)
    if meta and meta.get("complete"):
        return meta
    return None


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------
def collect_shard(
    spec: ShardSpec,
    *,
    engine: str | Path,
    data_dir: str | Path,
    shard_root: str | Path,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Run one shard and write ``labels.jsonl`` + pixels + ``shard.json``."""

    import numpy as np

    from touhou_rl.v400.gym.protocol import (
        parse_events_reply,
        parse_hazards_reply,
        parse_player_reply,
    )
    from touhou_rl.v400.gym.shm import SharedBlock
    from touhou_rl.v400.labels import (
        LABEL_VERSION,
        PLAYER_STATE_ALIVE,
        SCREEN_ORIGIN,
        decode_state,
        encode_occupancy,
        evaluate_future_danger,
        event_labels_v2,
        hazard_occupancy,
        hud_label,
        player_label,
        state_from_reply,
    )
    from touhou_rl.v400.labels.png import write_rgba_png

    engine = Path(engine)
    if not engine.is_file():
        raise SourceUnavailable(f"engine binary not found: {engine}")
    shard_root = Path(shard_root)
    attempt_dir = next_attempt_dir(shard_root)
    pixels_dir = attempt_dir / "pixels"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    pixels_dir.mkdir(parents=True, exist_ok=True)
    labels_path = attempt_dir / "labels.jsonl"
    labels_path.write_bytes(b"")

    engine_sha = sha256_file(engine)
    replay_env = {}
    if spec.source == "replay":
        replay_env = {
            "TH08_AUTOPLAY_REPLAY": str(spec.replay_path),
            "TH08_AUTOPLAY_STAGE": str(spec.stage),
        }
    shm_name = f"v400p2v2-{os.getpid()}-{int(time.time() * 1000) % 1000000}"
    client = ControlClient(
        engine,
        data_dir,
        role=spec.role,
        timeout=spec.timeout,
        stderr_path=log_path,
        extra_env=replay_env,
    )
    actions = Policy(
        "still" if spec.source == "replay" else spec.source,
        random.Random(spec.seed),
        hold=spec.hold,
        bomb_prob=spec.bomb_prob,
    )

    meta: dict[str, Any] = {
        "schema": "v400-p2-dataset-shard-2",
        "shard": spec.key,
        "spec": spec.as_dict(),
        "engine": str(engine),
        "engine_sha256": engine_sha,
        "data_dir": str(Path(data_dir).resolve()),
        "label_version": LABEL_VERSION,
        "screen_origin": list(SCREEN_ORIGIN),
        "grid": [96, 112, 4],
        "future_horizon": FUTURE_HORIZON,
        "attempt": attempt_dir.name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "complete": False,
        "replies": {},
        "channel_errors": {},
        "counts": {},
        "error": None,
    }

    events_error: str | None = None
    hazards_error: str | None = None
    player_error: str | None = None
    state_error: str | None = None
    availability: dict[str, dict[str, int]] = {
        name: {"ok": 0, "err": 0} for name in ("EVENTS", "HAZARDS", "PLAYER", "STATE")
    }
    pm = None
    pending: list[dict[str, Any]] = []
    written = 0
    recorded = 0
    warmup_steps = 0
    frame_mismatches = 0
    max_bullets = 0
    game_over_frame = None
    first_alive = False
    pixel_seconds = 0.0
    counts = {
        "hit": 0,
        "player_dying_started": 0,
        "effect_spawned": 0,
        "death_events": 0,
        "death_proxy": 0,
        "bomb_events": 0,
        "bomb_proxy": 0,
        "game_over": 0,
        "extend": 0,
        "alive_to_dying": 0,
        "hit_and_alive_to_dying": 0,
        "danger_positive": 0,
        "danger_negative": 0,
        "danger_null": 0,
        "occupancy_nonempty": 0,
        "occupancy_empty": 0,
    }
    label_writer = _LabelWriter(
        labels_path=labels_path,
        pixels_dir=pixels_dir,
        attempt_dir=attempt_dir,
        source=spec.source,
        stage=spec.stage,
        difficulty=spec.difficulty,
        seed=spec.seed,
        pixel_format=spec.pixel_format,
        engine_sha=engine_sha,
    )

    started = time.monotonic()
    try:
        client.start()
        meta["replies"]["hello"] = client.request("HELLO")
        meta["replies"]["debug"] = client.try_request("DEBUG")[0]
        meta["replies"]["shm"] = client.request(f"SHM /{shm_name}")
        readback = client.request("READBACK 1")
        if "readback=1" not in readback:
            raise EngineError(f"READBACK 1 refused: {readback!r}")
        meta["replies"]["readback"] = readback
        pm = SharedBlock(shm_name)
        if spec.source == "replay":
            meta["replies"]["replay"] = "autoplay-playback"
        else:
            reset_started = time.monotonic()
            meta["replies"]["reset"] = client.request(
                f"RESET {spec.stage} {spec.difficulty} 0 {spec.seed} 0", timeout=240.0
            )
            meta["reset_seconds"] = round(time.monotonic() - reset_started, 3)
            if "ready=0" in meta["replies"]["reset"]:
                raise EngineError(f"RESET did not become ready: {meta['replies']['reset']}")

        def read_frame(index: int, mask: int, record: bool) -> dict[str, Any] | None:
            nonlocal frame_mismatches, game_over_frame, max_bullets
            nonlocal events_error, hazards_error, player_error, state_error
            stepped = parse_reply(client.request(f"STEP {mask} 1"))
            if int(stepped.get("advanced", 0)) != 1:
                game_over_frame = index
                raise EngineError(f"STEP did not advance (chain exit): {stepped!r}")
            frame_number = int(stepped["frame"])
            header = pm.header()
            delta = frame_number - int(header["frame"])
            if delta != 1:
                frame_mismatches += 1
            t0 = time.monotonic()
            pixels = pm.pixels() if record else None
            pixel_cost = time.monotonic() - t0

            def note(name: str, error: str | None) -> None:
                if not record:
                    return
                availability[name]["err" if error is not None else "ok"] += 1

            events = None
            reply, error = client.try_request("EVENTS")
            if reply is not None:
                try:
                    events = parse_events_reply(reply)
                except Exception as exc:  # protocol error -> unavailable, not zero
                    error = f"parse: {exc}"
            note("EVENTS", error)
            if error is not None:
                events_error = error if events_error is None else events_error

            hazards = None
            reply, error = client.try_request("HAZARDS")
            if reply is not None:
                try:
                    hazards = parse_hazards_reply(reply)
                except Exception as exc:
                    error = f"parse: {exc}"
            note("HAZARDS", error)
            if error is not None:
                hazards_error = error if hazards_error is None else hazards_error
            if hazards is not None:
                max_bullets = max(max_bullets, int(hazards.bullet_count))

            player = None
            reply, error = client.try_request("PLAYER")
            if reply is not None:
                try:
                    player = parse_player_reply(reply)
                except Exception as exc:
                    error = f"parse: {exc}"
            note("PLAYER", error)
            if error is not None:
                player_error = error if player_error is None else player_error

            state = None
            reply, error = client.try_request("STATE")
            if reply is not None:
                try:
                    state = state_from_reply(reply)
                except Exception as exc:
                    error = f"parse: {exc}"
            note("STATE", error)
            if error is not None:
                state_error = error if state_error is None else state_error

            if not record:
                return {
                    "player_state": None if player is None else int(player.player_state)
                }
            return {
                "i": index,
                "mask": mask,
                "frame": frame_number,
                "shm_frame": int(header["frame"]),
                "pixel_stamp_delta": delta,
                "game_over": stepped.get("game_over") == "1",
                "events": events,
                "hazards": hazards,
                "player": player,
                "state": state,
                "pixels": pixels,
                "pixel_seconds": pixel_cost,
            }

        def flush_one() -> None:
            """Emit the oldest buffered frame once its future window is filled."""

            nonlocal written
            item = pending[0]
            successors = [row["hazards"] for row in pending[1 : 1 + FUTURE_HORIZON]]
            prev = pending_prev_state[0] if pending_prev_state else None
            payload = label_writer.emit(
                item,
                successors=successors,
                prev_state=prev,
                events_error=events_error,
                hazards_error=hazards_error,
                player_error=player_error,
                next_player_state=(
                    int(pending[1]["player"].player_state)
                    if len(pending) > 1 and pending[1]["player"] is not None
                    else None
                ),
                counts=counts,
            )
            pending_prev_state[0] = payload["state"]
            written += 1
            pending.pop(0)

        pending_prev_state: list[Any] = [None]

        total = spec.warmup + spec.frames
        for index in range(-spec.warmup, spec.frames):
            mask = actions.mask(index if index >= 0 else 0)
            if not first_alive:
                probe = read_frame(index, mask, False)
                warmup_steps += 1
                if probe is not None and probe["player_state"] == PLAYER_STATE_ALIVE:
                    first_alive = True
                continue
            item = read_frame(len(pending) + written, mask, True)
            assert item is not None
            pending.append(item)
            recorded += 1
            while len(pending) > FUTURE_HORIZON:
                flush_one()
            if recorded >= spec.frames:
                break
        while pending:
            flush_one()
        meta["replies"]["readback_final"] = client.request("READBACK")
        meta["complete"] = True
    except EngineError as exc:
        meta["error"] = str(exc)
    except SourceUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - keep partial evidence, never fake success
        meta["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if pm is not None:
            try:
                pm.close()
            except Exception:
                pass
        client.close()
        for path in Path("/dev/shm").glob(f"{shm_name}*"):
            try:
                path.unlink()
            except OSError:
                pass

    wall = time.monotonic() - started
    meta.update(
        {
            "frames_requested": spec.frames,
            "frames_recorded": recorded,
            "frames_written": written,
            "warmup_steps": warmup_steps,
            "warmup_to_first_alive": first_alive,
            "game_over_frame": game_over_frame,
            "shm_frame_mismatches": frame_mismatches,
            "max_bullets": max_bullets,
            "bombs_pressed_by_policy": actions.bombs_pressed,
            "channel_availability": availability,
            "channel_errors": {
                "EVENTS": events_error,
                "HAZARDS": hazards_error,
                "PLAYER": player_error,
                "STATE": state_error,
            },
            "counts": counts,
            "wall_seconds": round(wall, 3),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "labels_sha256": (
                sha256_file(labels_path) if labels_path.stat().st_size else None
            ),
            "labels_bytes": labels_path.stat().st_size,
        }
    )
    meta["complete"] = bool(meta["complete"] and written > 0 and meta["error"] is None)
    attempt_meta_path(attempt_dir).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


class _LabelWriter:
    """Writes one label line + one pixel file per emitted frame."""

    def __init__(
        self,
        *,
        labels_path: Path,
        pixels_dir: Path,
        attempt_dir: Path,
        source: str,
        stage: int,
        difficulty: int,
        seed: int,
        pixel_format: str,
        engine_sha: str,
    ) -> None:
        self.labels_path = labels_path
        self.pixels_dir = pixels_dir
        self.attempt_dir = attempt_dir
        self.source = source
        self.stage = stage
        self.difficulty = difficulty
        self.seed = seed
        self.pixel_format = pixel_format
        self.engine_sha = engine_sha

    def emit(
        self,
        item: dict[str, Any],
        *,
        successors: list[Any],
        prev_state: dict[str, Any] | None,
        events_error: str | None,
        hazards_error: str | None,
        player_error: str | None,
        next_player_state: int | None,
        counts: dict[str, int],
    ) -> dict[str, Any]:
        from touhou_rl.v400.labels import (
            LABEL_VERSION,
            encode_occupancy,
            evaluate_future_danger,
            event_labels_v2,
            hazard_occupancy,
            hud_label,
            player_label,
        )
        from touhou_rl.v400.labels.png import write_rgba_png

        player = item["player"]
        state = item["state"]
        events = item["events"]
        flags = None if events is None else int(events.frame_event_flags)
        labels = event_labels_v2(
            frame_event_flags=flags,
            player_state=None if player is None else int(player.player_state),
            next_player_state=next_player_state,
            state=state,
            prev_state=prev_state,
            step_game_over=item["game_over"],
            events_error=events_error,
        )

        if item["hazards"] is None:
            occupancy = None
            danger = {
                "would_die": None,
                "first_hit_offset": None,
                "min_gap": None,
                "per_horizon": {"4": None, "8": None, "16": None},
                "frames_considered": 0,
                "frames_with_hazard": 0,
                "applicable": False,
                "reason": f"hazards_channel_unavailable: {hazards_error}",
                "model": "engine-phase",
                "still": True,
            }
        else:
            occupancy = encode_occupancy(hazard_occupancy(item["hazards"]))
            if player is None:
                danger = {
                    "would_die": None,
                    "first_hit_offset": None,
                    "min_gap": None,
                    "per_horizon": {"4": None, "8": None, "16": None},
                    "frames_considered": 0,
                    "frames_with_hazard": 0,
                    "applicable": False,
                    "reason": f"player_channel_unavailable: {player_error}",
                    "model": "engine-phase",
                    "still": True,
                }
            else:
                danger = evaluate_future_danger(
                    player, [h for h in successors if h is not None], horizon=FUTURE_HORIZON
                ).as_dict()

        pixels = write_rgba_png(self.pixels_dir / f"frame-{item['i']:05d}.png", item["pixels"])
        pixels["file"] = f"pixels/frame-{item['i']:05d}.png"

        hud = hud_label(state or {})
        payload: dict[str, Any] = {
            "schema": "v400-p2-dataset-label-2",
            "shard": self.attempt_dir.parent.name,
            "attempt": self.attempt_dir.name,
            "source": self.source,
            "stage": self.stage,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "i": item["i"],
            "mask": item["mask"],
            "frame": item["frame"],
            "shm_frame": item["shm_frame"],
            "game_over": item["game_over"],
            "player": player_label(player),
            "hud": hud,
            "events": labels,
            "occupancy": occupancy,
            "future_danger": danger,
            "state": state,
            "pixels": pixels,
            "meta": {
                "label_version": LABEL_VERSION,
                "engine_sha256": self.engine_sha,
                "label_sources": (
                    "STATE+PLAYER+EVENTS+HAZARDS; pixels stored, never read"
                    if item["hazards"] is not None
                    else "STATE+PLAYER+EVENTS (HAZARDS unavailable in this mode); "
                    "pixels stored, never read"
                ),
            },
        }
        with open(self.labels_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"
            )

        # running counters, so the shard meta can be read without the JSONL
        if labels["hit"]["value"]:
            counts["hit"] += 1
        if labels["player_dying_started"]["value"]:
            counts["player_dying_started"] += 1
        if labels["effect_spawned"]["value"]:
            counts["effect_spawned"] += 1
        if labels["death"]["value"]:
            counts["death_events"] += 1
        if labels["death"]["proxy"]["value"]:
            counts["death_proxy"] += 1
        if labels["death"]["proxy"]["components"]["alive_to_dying"]:
            counts["alive_to_dying"] += 1
        if labels["bomb"]["value"]:
            counts["bomb_events"] += 1
        if labels["bomb"]["proxy"]["value"]:
            counts["bomb_proxy"] += 1
        if labels["game_over"]["value"]:
            counts["game_over"] += 1
        if labels["extend"]["value"]:
            counts["extend"] += 1
        if labels["hit"]["value"] and labels["death"]["proxy"]["components"]["alive_to_dying"]:
            counts["hit_and_alive_to_dying"] += 1
        if danger.get("would_die") is True:
            counts["danger_positive"] += 1
        elif danger.get("would_die") is False:
            counts["danger_negative"] += 1
        else:
            counts["danger_null"] += 1
        if occupancy is not None:
            if int(occupancy.get("occupied_cells", 0)) > 0:
                counts["occupancy_nonempty"] += 1
            else:
                counts["occupancy_empty"] += 1
        return payload


# ---------------------------------------------------------------------------
# manifest / resume
# ---------------------------------------------------------------------------
def scan_manifest(out_dir: str | Path) -> dict[str, Any]:
    """Aggregate every shard attempt under ``out_dir`` into one manifest."""

    from touhou_rl.v400.labels import LABEL_VERSION

    out_dir = Path(out_dir)
    shards_root = out_dir / "shards"
    shards: list[dict[str, Any]] = []
    total_frames = 0
    total_complete_frames = 0
    engine_shas: dict[str, int] = {}
    for shard_root in sorted(path for path in shards_root.glob("*") if path.is_dir()):
        attempts = sorted(
            (path for path in shard_root.glob("attempt-*") if path.is_dir()),
            key=lambda path: path.name,
        )
        rows = []
        for attempt in attempts:
            meta = read_attempt_meta(attempt)
            labels = attempt / "labels.jsonl"
            rows.append(
                {
                    "attempt": attempt.name,
                    "complete": bool(meta and meta.get("complete")),
                    "frames_written": None if not meta else meta.get("frames_written"),
                    "frames_requested": None if not meta else meta.get("frames_requested"),
                    "error": None if not meta else meta.get("error"),
                    "labels_bytes": labels.stat().st_size if labels.is_file() else 0,
                    "labels_sha256": None if not meta else meta.get("labels_sha256"),
                    "counts": None if not meta else meta.get("counts"),
                    "channel_errors": None if not meta else meta.get("channel_errors"),
                    "wall_seconds": None if not meta else meta.get("wall_seconds"),
                }
            )
            if meta and meta.get("engine_sha256"):
                sha = meta["engine_sha256"]
                engine_shas[sha] = engine_shas.get(sha, 0) + 1
            if meta:
                total_frames += int(meta.get("frames_written") or 0)
                if meta.get("complete"):
                    total_complete_frames += int(meta.get("frames_written") or 0)
        selected = None
        for row in reversed(rows):
            if row["complete"]:
                selected = row["attempt"]
                break
        shards.append(
            {
                "shard": shard_root.name,
                "complete": bool(selected),
                "selected_attempt": selected,
                "attempts": rows,
            }
        )
    manifest = {
        "schema": "v400-p2-dataset-manifest-2",
        "out_dir": str(out_dir),
        "label_version": LABEL_VERSION,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine_sha256": engine_shas,
        "shard_count": len(shards),
        "complete_shards": sum(1 for shard in shards if shard["complete"]),
        "frames_written_all_attempts": total_frames,
        "frames_written_complete_shards": total_complete_frames,
        "shards": shards,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def complete_label_files(out_dir: str | Path) -> list[Path]:
    """``labels.jsonl`` of every shard's selected (complete) attempt."""

    out_dir = Path(out_dir)
    files: list[Path] = []
    for shard_root in sorted(path for path in (out_dir / "shards").glob("*") if path.is_dir()):
        attempt = latest_attempt(shard_root)
        if attempt is None:
            continue
        meta = read_attempt_meta(attempt)
        if meta and meta.get("complete") and (attempt / "labels.jsonl").is_file():
            files.append(attempt / "labels.jsonl")
    return files
