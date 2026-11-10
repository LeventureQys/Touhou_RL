#!/usr/bin/env python3
"""P2-WP3 bulk collector: compact observation pixels + reward labels (EXP-P2-003).

Why this exists
---------------
``tools/rl/v400_p2_dataset.py`` stores one lossless 640x480 RGBA PNG per frame
(~360 KB).  At the volume the G2 gate needs (>= 300 k frames) that is > 100 GB of
pixels, so it cannot be scaled by repeating that tool.  This collector is the
P2-WP3 path: it keeps **only the downsampled observation** the detector will
actually see, plus the numeric label rows.

Per frame, three fixed-size uint8 planes are appended to three flat files:

* ``play96.u8``  — P2 §3.3 grid, 4x4 block mean of the 448x384 playfield at
  luminance → ``96 x 112`` (10 752 B).  This is "the 1/4 grid" of the spec.
* ``play64.u8``  — 7x6 block mean of the same playfield → ``64 x 64`` (4 096 B),
  stored natively so the P2-WP4 resolution comparison does not have to resample
  the 96-wide grid.
* ``hud.u8``     — the three numeric HUD bands (score / lives / bombs) cropped at
  **native** resolution from ``canvas.json``'s ``Gui.cpp`` anchors: screen rows
  48-72, 80-96, 96-112, columns 480-640 → ``56 x 160`` (8 960 B).  Half-resolution
  HUD was measured unreadable (see EXP-P2-003 "why the HUD is not downsampled"):
  4x4 digits collapse into 2x2 blobs and the lives/bombs icons stop being
  countable, so a >= 99.5 % numeric-read gate could not even in principle be met
  from a downsampled HUD.

Byte cost: 23 808 B/frame → 300 k frames = 6.84 GiB.  Labels are a compact JSONL
row (~170 B) that keeps the raw ``EVENTS`` word and the decoded ``STATE`` values;
nothing is inferred from pixels and pixels are never read to make a label.

Loop shape
----------
One *shard* is one engine process: ``SHM`` + ``READBACK 1``, ``RESET``, warm-up
to the first ``ALIVE`` frame, then ``frames`` recorded frames of
``STEP`` + ``EVENTS`` + ``STATE``.  Three round trips per frame (the P2-WP2
collector also dialled ``PLAYER`` and ``HAZARDS``; both are redundant here —
``STATE`` already carries lives/bombs/score/position and the reward labels come
from ``EVENTS``).

A shard ends on: the frame budget, a step that no longer advances (game over →
continue screen), or the wall-clock deadline.  Frames written before that are
valid data and are kept; ``meta.json`` records which of those happened.

Parallelism
-----------
``run`` starts N ``worker`` subprocesses, each with a private ``--data-dir``
(own ``score.dat``/``th08.cfg`` so instances never share writable game state),
a private ``/dev/shm`` segment and a private shard list.  All workers inherit one
``DISPLAY`` (a single Xvfb) — measured cheaper than 16 X servers because llvmpipe
renders inside the client process; the X server only carries GLX.

Commands
--------
``prepare``  create the per-instance data dirs (symlinks to the read-only game
             assets, private copies of the writable ones).  Never deletes.
``worker``   one process, its own shard list (resumable).
``run``      the orchestration above.
``stats``    frame/positive/byte audit + the G2 data-volume gap.
``index``    ``index.json`` for the detector's stage+seed split.

All paths in this module are Linux paths: it is meant to run inside the WSL
guest, where the engine and ``/dev/shm`` live.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Bumped when a stored plane or label row changes meaning.
DATASET_SCHEMA = "v400-p2det-shard-1"

#: Plane geometry.  ``play96``: 384x448 playfield / 4; ``play64``: /6 and /7;
#: ``hud``: the three numeric bands stacked (see ``hud_bands``).
PLAY96_SHAPE = (112, 96)
PLAY64_SHAPE = (64, 64)

#: HUD bands actually used for the numeric heads: (row0, row1) in screen pixels.
#: Derived from ``touhou_rl.v400.labels.spec.HUD_LAYOUT`` in :func:`hud_bands`.
HUD_BAND_KEYS = ("score", "lives", "bombs")
HUD_X0, HUD_X1 = 480, 640

#: Event bits (``touhou_rl.v400.labels.events.EVENT_BITS``), repeated here so a
#: stored row can be decoded without importing the labels package.
EVENT_BIT_BOMB = 0x0001
EVENT_BIT_HIT = 0x0002
EVENT_BIT_DEATH = 0x0004
EVENT_BIT_DYING = 0x0200
EVENT_BIT_EFFECT = 0x0400

#: ``p2-v2-1`` is the label meaning this collector records (same ``EVENTS`` bits
#: and the same "state value, not OCR guess" HUD semantics as P2-WP2).
LABEL_VERSION = "p2-v2-1"


def hud_bands() -> list[tuple[int, int]]:
    """``[(row0, row1)]`` of the score/lives/bombs bands, from the frozen layout."""

    from touhou_rl.v400.labels.spec import HUD_LAYOUT

    bands = []
    for key in HUD_BAND_KEYS:
        _, y0, _, y1 = HUD_LAYOUT[key]["band"]
        bands.append((int(y0), int(y1)))
    return bands


def hud_shape() -> tuple[int, int]:
    bands = hud_bands()
    height = sum(y1 - y0 for y0, y1 in bands)
    return (height, HUD_X1 - HUD_X0)


def plane_shapes() -> dict[str, tuple[int, int]]:
    return {
        "play96": PLAY96_SHAPE,
        "play64": PLAY64_SHAPE,
        "hud": hud_shape(),
    }


def frame_bytes() -> dict[str, int]:
    return {name: h * w for name, (h, w) in plane_shapes().items()}


# ---------------------------------------------------------------------------
# pixel encoding
# ---------------------------------------------------------------------------
def encode_frame(pixels, *, playfield_origin: tuple[int, int], playfield: tuple[int, int]):
    """640x480 RGBA -> ``(play96, play64, hud)`` uint8 arrays.

    Luminance is ``(77 R + 151 G + 28 B) >> 8`` (the integer BT.601 weights), so
    the planes are a documented, exactly reproducible function of the readback
    bytes.  Block values are integer means (``sum // n``) — no float rounding and
    no dithering.
    """

    import numpy as np

    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError(f"pixels must be uint8 HxWx4, got {pixels.dtype} {pixels.shape}")
    x0, y0 = playfield_origin
    width, height = playfield
    crop = pixels[y0 : y0 + height, x0 : x0 + width]
    if crop.shape[0] != height or crop.shape[1] != width:
        raise ValueError(
            f"readback {pixels.shape} does not contain the {width}x{height} playfield"
        )
    red = crop[:, :, 0].astype(np.uint16)
    green = crop[:, :, 1].astype(np.uint16)
    blue = crop[:, :, 2].astype(np.uint16)
    lum = ((red * 77 + green * 151 + blue * 28) >> 8).astype(np.uint32)

    oh, ow = PLAY96_SHAPE
    bh, bw = height // oh, width // ow
    play96 = (lum.reshape(oh, bh, ow, bw).sum(axis=(1, 3)) // (bh * bw)).astype(np.uint8)

    oh64, ow64 = PLAY64_SHAPE
    bh64, bw64 = height // oh64, width // ow64
    play64 = (lum.reshape(oh64, bh64, ow64, bw64).sum(axis=(1, 3)) // (bh64 * bw64)).astype(
        np.uint8
    )

    hud_rows = [
        pixels[y0b:y1b, HUD_X0:HUD_X1, 1] for (y0b, y1b) in hud_bands()
    ]
    hud = np.concatenate(hud_rows, axis=0)
    return play96, play64, hud


# ---------------------------------------------------------------------------
# shard storage
# ---------------------------------------------------------------------------
@dataclass
class ShardSpec:
    """One bounded engine run; ``key`` is the resume and split identity."""

    source: str
    stage: int
    difficulty: int
    seed: int
    frames: int
    hold: int = 6
    bomb_prob: float = 0.02
    warmup: int = 900
    lp_threads: int = 4
    tag: str = ""
    #: Poll the read-only ``HUD`` command on every recorded frame (needs the
    #: ``rl_score`` build).  Off by default: the P2 corpus is ``EVENTS``/``STATE``
    #: only, and turning this on costs one extra round trip per frame.  The G2
    #: death/game-over top-up turns it on because ``gameOverFlag`` and
    #: ``displayStale`` only exist on ``HUD`` (EXP-P2-005).
    hud: bool = False

    @property
    def key(self) -> str:
        base = f"{self.source}-s{self.stage}d{self.difficulty}-seed{self.seed}-n{self.frames}"
        if self.bomb_prob > 0.0:
            base += f"-bp{self.bomb_prob:g}"
        if self.tag:
            base += f"-{self.tag}"
        if self.hud:
            base += "-hud"
        return base

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "stage": self.stage,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "frames": self.frames,
            "hold": self.hold,
            "bomb_prob": self.bomb_prob,
            "warmup": self.warmup,
            "lp_threads": self.lp_threads,
            "tag": self.tag,
            "hud": self.hud,
            "key": self.key,
        }


def plane_path(shard_dir: Path, name: str) -> Path:
    return Path(shard_dir) / f"{name}.u8"


def open_plane(shard_dir: Path, name: str, frames: int):
    """``np.memmap`` view ``(frames, H, W)`` of one stored plane (read only)."""

    import numpy as np

    height, width = plane_shapes()[name]
    return np.memmap(
        plane_path(shard_dir, name), dtype=np.uint8, mode="r", shape=(frames, height, width)
    )


def read_labels(shard_dir: Path) -> list[dict[str, Any]]:
    path = Path(shard_dir) / "labels.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                # A torn last line from a killed worker: keep every whole row
                # that came before it, never invent the partial one.
                break
    return rows


def shard_frame_count(shard_dir: Path) -> int:
    """Frames readable from a shard: whole label rows AND whole pixel frames."""

    rows = len(read_labels(Path(shard_dir)))
    if rows == 0:
        return 0
    for name, size in frame_bytes().items():
        path = plane_path(Path(shard_dir), name)
        if not path.is_file():
            return 0
        rows = min(rows, path.stat().st_size // size)
    return rows


def shard_complete(shard_root: Path) -> dict[str, Any] | None:
    attempt = latest_attempt(shard_root)
    if attempt is None:
        return None
    meta_path = attempt / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return meta if meta.get("complete") else None


def latest_attempt(shard_root: Path) -> Path | None:
    attempts = sorted(
        (p for p in Path(shard_root).glob("attempt-*") if p.is_dir()), key=lambda p: p.name
    )
    return attempts[-1] if attempts else None


def next_attempt_dir(shard_root: Path) -> Path:
    """``shards/<key>/attempt-NN`` with the next free NN.  Never overwrites."""

    existing = sorted(
        int(p.name.split("-", 1)[1])
        for p in Path(shard_root).glob("attempt-*")
        if p.is_dir() and p.name.split("-", 1)[1].isdigit()
    )
    number = (existing[-1] + 1) if existing else 1
    return Path(shard_root) / f"attempt-{number:02d}"


# ---------------------------------------------------------------------------
# one shard
# ---------------------------------------------------------------------------
def run_shard(
    spec: ShardSpec,
    *,
    engine: str | Path,
    data_dir: str | Path,
    shard_root: str | Path,
    log_path: Path | None = None,
    deadline_epoch: float | None = None,
    shm_name: str | None = None,
) -> dict[str, Any]:
    """Drive one engine process and append frames to ``attempt-NN``."""

    import numpy as np

    from touhou_rl.v400.gym.protocol import parse_events_reply, parse_hud_reply
    from touhou_rl.v400.gym.shm import SharedBlock
    from touhou_rl.v400.labels.runner import ControlClient, sha256_file
    from touhou_rl.v400.labels.spec import (
        PLAYFIELD_HEIGHT,
        PLAYFIELD_WIDTH,
        SCREEN_ORIGIN,
    )
    from touhou_rl.v400.labels.state import state_from_reply

    engine = Path(engine)
    if not engine.is_file():
        raise SystemExit(f"engine binary not found: {engine}")
    shard_root = Path(shard_root)
    attempt = next_attempt_dir(shard_root)
    attempt.mkdir(parents=True, exist_ok=True)
    labels_path = attempt / "labels.jsonl"
    labels_path.write_bytes(b"")
    handles = {name: open(plane_path(attempt, name), "wb") for name in frame_bytes()}
    label_handle = open(labels_path, "a", encoding="utf-8")

    engine_sha = sha256_file(engine)
    shm = shm_name or f"p2det-{os.getpid()}-{int(time.time() * 1000) % 10_000_000}"
    client = ControlClient(
        engine,
        data_dir,
        role="env",
        timeout=120.0,
        stderr_path=log_path,
        extra_env={"LP_NUM_THREADS": str(int(spec.lp_threads))},
    )
    policy = build_policy(spec.source, spec.seed, hold=spec.hold, bomb_prob=spec.bomb_prob)

    counts = {
        "death": 0,
        "bomb": 0,
        "hit": 0,
        "dying_started": 0,
        "effect_spawned": 0,
        "game_over_frames": 0,
        "game_over_hud_frames": 0,
        "hud_missing": 0,
        "events_missing": 0,
        "state_missing": 0,
        "score_up": 0,
        "score_same": 0,
        "score_down": 0,
    }
    meta: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "shard": spec.key,
        "spec": spec.as_dict(),
        "attempt": attempt.name,
        "engine": str(engine),
        "engine_sha256": engine_sha,
        "data_dir": str(Path(data_dir)),
        "label_version": LABEL_VERSION,
        "planes": {name: list(shape) for name, shape in plane_shapes().items()},
        "playfield": {
            "origin": list(SCREEN_ORIGIN),
            "width": PLAYFIELD_WIDTH,
            "height": PLAYFIELD_HEIGHT,
            "channel": "BT.601 integer luminance (77R+151G+28B)>>8",
            "block": {"play96": [4, 4], "play64": [7, 6]},
            "hud": {
                "bands": [list(band) for band in hud_bands()],
                "keys": list(HUD_BAND_KEYS),
                "columns": [HUD_X0, HUD_X1],
                "channel": "green (native resolution, not downsampled)",
            },
        },
        "shm_name": shm,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "deadline_epoch": deadline_epoch,
        "complete": False,
        "end_reason": None,
        "error": None,
        "warmup_steps": 0,
        "frames_written": 0,
        "counts": counts,
    }
    written = 0
    started = time.monotonic()
    pixels = None
    prev_state = None
    max_requests = 0
    pm = None
    try:
        client.start()
        client.request("HELLO")
        client.request(f"SHM /{shm}")
        readback = client.request("READBACK 1")
        if "readback=1" not in readback:
            raise RuntimeError(f"READBACK 1 refused: {readback!r}")
        pm = SharedBlock(shm)
        reset = client.request(f"RESET {spec.stage} {spec.difficulty} 0 {spec.seed} 0",
                              timeout=240.0)
        if "ready=0" in reset:
            raise RuntimeError(f"RESET did not become ready: {reset}")

        # --- warm-up: step (without recording) until the player is ALIVE -----
        warm = 0
        for _ in range(int(spec.warmup)):
            client.request("STEP 1 1")
            warm += 1
            reply, error = client.try_request("STATE")
            if reply is None:
                continue
            state = state_from_reply(reply)
            if int(state["player_state"]) == 0:  # PLAYER_STATE_ALIVE
                break
            if deadline_epoch is not None and time.time() > deadline_epoch:
                break
        meta["warmup_steps"] = warm

        while True:
            if deadline_epoch is not None and time.time() > deadline_epoch:
                meta["end_reason"] = "deadline"
                break
            if written >= spec.frames:
                meta["end_reason"] = "frames"
                break
            index = written
            mask = policy.mask(index)
            stepped = client.request(f"STEP {mask} 1")
            parts = dict(
                part.split("=", 1) for part in stepped.split()[1:] if "=" in part
            )
            if int(parts.get("advanced", 0)) != 1:
                meta["end_reason"] = "step_not_advanced"
                break
            frame = int(parts.get("frame", -1))
            pixels = pm.pixels()

            events_reply, events_error = client.try_request("EVENTS")
            flags = None
            if events_reply is not None:
                try:
                    flags = int(parse_events_reply(events_reply).frame_event_flags)
                except Exception:  # noqa: BLE001 - unparsable -> recorded, not faked
                    flags = None
            state_reply, state_error = client.try_request("STATE")
            state = None
            if state_reply is not None:
                try:
                    state = state_from_reply(state_reply)
                except Exception:  # noqa: BLE001
                    state = None
            hud_state = None
            if spec.hud:
                hud_reply, _ = client.try_request("HUD")
                if hud_reply is not None:
                    try:
                        hud_state = parse_hud_reply(hud_reply)
                    except Exception:  # noqa: BLE001 - recorded, never faked
                        hud_state = None
                counts["hud_missing"] += int(hud_state is None)

            play96, play64, hud = encode_frame(
                pixels,
                playfield_origin=SCREEN_ORIGIN,
                playfield=(PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT),
            )
            handles["play96"].write(play96.tobytes())
            handles["play64"].write(play64.tobytes())
            handles["hud"].write(hud.tobytes())

            row: dict[str, Any] = {
                "i": index,
                "f": frame,
                "go": 1 if parts.get("game_over") == "1" else 0,
                "ev": flags,
                "msk": mask,
            }
            if state is not None:
                row.update(
                    {
                        "li": int(state["lives_remaining"]),
                        "bo": int(state["bombs_remaining"]),
                        "sc": int(state["score"]),
                        "gr": int(state["graze"]),
                        "po": int(state["point_items_collected"]),
                        "ti": int(state["time_orbs"]),
                        "pw": int(state["player_power"]),
                        "dx": int(state["deaths"]),
                        "px": round(float(state["player_x"]), 1),
                        "py": round(float(state["player_y"]), 1),
                        "ps": int(state["player_state"]),
                    }
                )
            else:
                counts["state_missing"] += 1
            if hud_state is not None:
                row.update(
                    {
                        "hli": int(hud_state.lives),
                        "hbo": int(hud_state.bombs),
                        "hds": int(hud_state.display_score),
                        "hgs": int(hud_state.globals_score),
                        "hgo": int(hud_state.game_over),
                        "hage": int(hud_state.display_age),
                        "hstale": int(hud_state.display_stale if hud_state.display_stale is not None
                                      else -1),
                    }
                )
                if int(hud_state.game_over):
                    counts["game_over_hud_frames"] += 1
            if flags is None:
                counts["events_missing"] += 1
                row.update({"death": None, "bomb": None, "hit": None, "dying": None})
            else:
                death = bool(flags & EVENT_BIT_DEATH)
                bomb = bool(flags & EVENT_BIT_BOMB)
                hit = bool(flags & EVENT_BIT_HIT)
                dying = bool(flags & EVENT_BIT_DYING)
                row.update({"death": death, "bomb": bomb, "hit": hit, "dying": dying})
                counts["death"] += int(death)
                counts["bomb"] += int(bomb)
                counts["hit"] += int(hit)
                counts["dying_started"] += int(dying)
                counts["effect_spawned"] += int(bool(flags & EVENT_BIT_EFFECT))
            if row["go"]:
                counts["game_over_frames"] += 1
            if state is not None and prev_state is not None:
                delta = int(state["score"]) - prev_state
                if delta > 0:
                    counts["score_up"] += 1
                elif delta < 0:
                    counts["score_down"] += 1
                else:
                    counts["score_same"] += 1
            if state is not None:
                prev_state = int(state["score"])
            else:
                prev_state = None

            label_handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
            written += 1
            if written % 200 == 0:
                label_handle.flush()
                meta["frames_written"] = written
                meta["counts"] = dict(counts)
                meta["elapsed_seconds"] = round(time.monotonic() - started, 3)
                (attempt / "meta.json").write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
        meta["complete"] = True
    except Exception as exc:  # noqa: BLE001 - partial evidence beats a fake success
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["end_reason"] = meta["end_reason"] or "engine_error"
    finally:
        try:
            label_handle.flush()
            label_handle.close()
        except OSError:
            pass
        for handle in handles.values():
            try:
                handle.close()
            except OSError:
                pass
        if pm is not None:
            try:
                pm.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            max_requests = client.requests
        except Exception:  # noqa: BLE001
            max_requests = 0
        client.close()
        for path in Path("/dev/shm").glob(f"{shm}*"):
            try:
                path.unlink()
            except OSError:
                pass

    wall = time.monotonic() - started
    meta.update(
        {
            "frames_written": written,
            "counts": counts,
            "requests": max_requests,
            "wall_seconds": round(wall, 3),
            "fps": round(written / wall, 2) if wall > 0 else None,
            "bytes": {name: plane_path(attempt, name).stat().st_size for name in frame_bytes()},
            "labels_bytes": labels_path.stat().st_size,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    meta["complete"] = bool(meta["complete"] and written > 0 and meta["error"] is None)
    (attempt / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


# ---------------------------------------------------------------------------
# plan / instances
# ---------------------------------------------------------------------------
#: (stage, difficulty) pairs the shard plan cycles through.  Spread over stages
#: 0-6 and difficulties 1-3 so a stage+seed split has more than one level to
#: hold out, and so the death rate is not a property of one level.
STAGE_PLAN: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 2),
    (3, 3),
    (4, 3),
    (2, 1),
    (5, 3),
    (3, 1),
    (0, 3),
    (6, 2),
    (1, 1),
    (4, 1),
)

#: Policy cycle.  ``still`` (shoot only, never move) is the high-death source and
#: gets one third of the shards; ``random`` carries the rest.
POLICY_PLAN: tuple[str, ...] = ("random", "random", "still")

#: Collection-only policy that the P2-WP2 collector does not have, added because
#: ``random``/``still`` measured far too few deaths to reach the G2 positive
#: count at 300 k frames (EXP-P2-003).  ``dive`` flies to the top edge of the
#: playfield and stays there, where enemies spawn and bullets are emitted, so a
#: death happens within a few hundred frames.  It is a *data-collection*
#: parameter (like ``bomb_prob``), recorded in every shard's ``spec.source``, and
#: never a claim about how a trained policy behaves.
POLICY_NAMES = ("random", "still", "dive")


class TopDivePolicy:
    """Hold ``UP`` (+ ``SHOOT``) so the player parks at the top of the playfield."""

    def __init__(self, rng, *, drift_every: int = 0) -> None:
        self.rng = rng
        self.drift_every = int(drift_every)
        self.bombs_pressed = 0

    def mask(self, index: int) -> int:
        from touhou_rl.v400.labels.runner import (
            BUTTON_LEFT,
            BUTTON_RIGHT,
            BUTTON_SHOOT,
            BUTTON_UP,
        )

        mask = BUTTON_SHOOT | BUTTON_UP
        if self.drift_every > 0 and index % self.drift_every == 0:
            mask |= self.rng.choice((0, BUTTON_LEFT, BUTTON_RIGHT))
        return mask


def build_policy(name: str, seed: int, *, hold: int = 6, bomb_prob: float = 0.0):
    """Policy factory.  ``random``/``still`` are the frozen P2-WP2 policies."""

    import random

    from touhou_rl.v400.labels.runner import Policy

    if name in ("random", "still"):
        return Policy(name, random.Random(seed), hold=hold, bomb_prob=bomb_prob)
    if name == "dive":
        return TopDivePolicy(random.Random(seed), drift_every=hold * 4)
    raise ValueError(f"unknown policy {name!r}; known: {POLICY_NAMES}")


def parse_stage_plan(text: str | None) -> tuple[tuple[int, int], ...]:
    """``"3:3,3:1"`` -> ``((3, 3), (3, 1))``; ``None`` keeps :data:`STAGE_PLAN`.

    Top-up runs only: the P2 campaign's stage cycle is a property of the main
    collector, so the override is opt-in and never changes the default.
    """

    if text is None or not str(text).strip():
        return STAGE_PLAN
    plan: list[tuple[int, int]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            stage_text, difficulty_text = item.split(":", 1)
            plan.append((int(stage_text), int(difficulty_text)))
        except ValueError as exc:
            raise ValueError(f"bad --stage-plan item {item!r} (want 'stage:difficulty')") from exc
    if not plan:
        raise ValueError("--stage-plan resolved to an empty plan")
    return tuple(plan)


def plan_specs(
    *,
    total_frames: int,
    frames_per_shard: int,
    workers: int,
    seed_base: int = 1000,
    hold: int = 6,
    bomb_prob: float = 0.02,
    warmup: int = 900,
    lp_threads: int = 4,
    policy_plan: Iterable[str] = POLICY_PLAN,
    stage_plan: Iterable[tuple[int, int]] = STAGE_PLAN,
    hud: bool = False,
) -> list[tuple[int, ShardSpec]]:
    """``[(worker_index, spec)]`` covering ``total_frames`` with slack.

    Slack is deliberate: shards end early on game over, so each worker is given
    ~1.6x the shard count it needs and stops on its own frame budget.
    """

    if workers < 1:
        raise ValueError("workers must be >= 1")
    if frames_per_shard < 1:
        raise ValueError("frames_per_shard must be >= 1")
    policies = tuple(policy_plan)
    if not policies:
        raise ValueError("policy plan must not be empty")
    unknown = [name for name in policies if name not in POLICY_NAMES]
    if unknown:
        raise ValueError(f"unknown policies in plan: {unknown}; known: {POLICY_NAMES}")
    per_worker = -(-total_frames // workers)
    needed = -(-per_worker // frames_per_shard)
    planned = max(needed + 3, int(needed * 1.6))
    stages = tuple((int(stage), int(difficulty)) for stage, difficulty in stage_plan)
    if not stages:
        raise ValueError("stage plan must not be empty")
    specs: list[tuple[int, ShardSpec]] = []
    for index in range(planned * workers):
        worker = index % workers
        round_index = index // workers
        stage, difficulty = stages[round_index % len(stages)]
        source = policies[round_index % len(policies)]
        spec = ShardSpec(
            source=source,
            stage=stage,
            difficulty=difficulty,
            seed=seed_base + index,
            frames=frames_per_shard,
            hold=hold,
            bomb_prob=bomb_prob if source == "random" else 0.0,
            warmup=warmup,
            lp_threads=lp_threads,
            tag=f"w{worker:02d}",
            hud=hud,
        )
        specs.append((worker, spec))
    return specs


def prepare_instances(*, data_root: str | Path, instances: int,
                      assets: str | Path = "/root/v400-data/game",
                      template: str | Path = "/root/v400-data/p2/game") -> list[str]:
    """Create ``<data_root>/inst-XX`` once; idempotent, never deletes."""

    data_root = Path(data_root)
    assets = Path(assets)
    template = Path(template)
    created = []
    for index in range(instances):
        target = data_root / f"inst-{index:02d}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "backup").mkdir(exist_ok=True)
        for asset in ("th08.dat", "thbgm.dat"):
            link = target / asset
            source = assets / asset
            if not link.exists():
                if not source.exists():
                    raise SystemExit(f"missing game asset: {source}")
                link.symlink_to(source)
        for writable in ("th08.cfg", "score.dat"):
            copy = target / writable
            if not copy.exists():
                source = template / writable
                if source.exists():
                    shutil.copy2(source, copy)
        created.append(str(target))
    return created


# ---------------------------------------------------------------------------
# worker / run
# ---------------------------------------------------------------------------
def worker_command(args) -> int:
    specs_all = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    mine = [row for row in specs_all if int(row["worker"]) == int(args.index)]
    data_dir = Path(args.data_root) / f"inst-{int(args.index):02d}"
    out_root = Path(args.out_dir)
    budget = int(args.frames)
    deadline = time.time() + float(args.max_seconds) if args.max_seconds else None
    done = 0
    results = []
    for row in mine:
        if done >= budget:
            break
        if deadline is not None and time.time() > deadline:
            break
        spec_fields = {k: v for k, v in row["spec"].items() if k != "key"}
        spec = ShardSpec(**spec_fields)
        spec.frames = min(spec.frames, budget - done)
        shard_root = out_root / "shards" / spec.key
        existing = shard_complete(shard_root)
        if existing is not None:
            results.append({"shard": spec.key, "skipped": True,
                            "frames": existing.get("frames_written")})
            done += int(existing.get("frames_written") or 0)
            continue
        log_path = out_root / "logs" / f"{spec.key}-{int(time.time())}.engine.stderr.log"
        meta = run_shard(
            spec,
            engine=args.engine,
            data_dir=data_dir,
            shard_root=shard_root,
            log_path=log_path,
            deadline_epoch=deadline,
        )
        done += int(meta.get("frames_written") or 0)
        results.append(
            {
                "shard": spec.key,
                "skipped": False,
                "frames": meta.get("frames_written"),
                "counts": meta.get("counts"),
                "fps": meta.get("fps"),
                "end_reason": meta.get("end_reason"),
                "error": meta.get("error"),
            }
        )
        print(json.dumps({"worker": int(args.index), **results[-1]}, ensure_ascii=False),
              flush=True)
    print(
        json.dumps(
            {"worker": int(args.index), "frames_total": done, "shards": len(results)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if done > 0 else 5


def progress_totals(out_dir: str | Path) -> dict[str, Any]:
    """Cheap progress: sum the per-attempt ``meta.json`` counters, no label reads.

    Used by the orchestrator's periodic line; the authoritative numbers come from
    :func:`scan` at the end (which also verifies whole label rows against whole
    pixel frames).
    """

    frames = 0
    counts = {"death": 0, "bomb": 0, "hit": 0}
    pixel_bytes = 0
    for shard_root in sorted(p for p in (Path(out_dir) / "shards").glob("*") if p.is_dir()) \
            if (Path(out_dir) / "shards").is_dir() else []:
        for attempt in sorted(p for p in shard_root.glob("attempt-*") if p.is_dir()):
            try:
                meta = json.loads((attempt / "meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            frames += int(meta.get("frames_written") or 0)
            for key in counts:
                counts[key] += int((meta.get("counts") or {}).get(key) or 0)
            for size in (meta.get("bytes") or {}).values():
                pixel_bytes += int(size or 0)
    return {"frames": frames, "counts": counts, "pixel_bytes": pixel_bytes}


def run_command(args) -> int:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if getattr(args, "max_frames", None):
        args.frames_per_shard = int(args.max_frames)
    specs = plan_specs(
        total_frames=args.frames_total,
        frames_per_shard=args.frames_per_shard,
        workers=args.instances,
        seed_base=args.seed_base,
        hold=args.hold,
        bomb_prob=args.bomb_prob,
        warmup=args.warmup,
        lp_threads=args.lp_threads,
        policy_plan=tuple(item for item in args.policy_plan.split(",") if item),
        stage_plan=parse_stage_plan(args.stage_plan),
        hud=bool(getattr(args, "hud", False)),
    )
    plan_path = out_root / "plan.json"
    plan_path.write_text(
        json.dumps([{"worker": w, "spec": s.as_dict()} for w, s in specs], indent=2) + "\n",
        encoding="utf-8",
    )
    prepare_instances(data_root=args.data_root, instances=args.instances,
                     assets=args.assets, template=args.template)
    per_worker = -(-args.frames_total // args.instances)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    procs = []
    started = time.time()
    for index in range(args.instances):
        log = open(out_root / "logs" / f"worker-{index:02d}.log", "ab", buffering=0)
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--engine", str(args.engine),
            "--data-root", str(args.data_root),
            "--out-dir", str(out_root),
            "--plan", str(plan_path),
            "--index", str(index),
            "--frames", str(per_worker),
            "--max-seconds", str(args.max_seconds),
        ]
        procs.append((index, subprocess.Popen(argv, stdout=log, stderr=log), log))

    printed = 0.0
    exit_codes = {}
    while True:
        alive = [p for _, p, _ in procs if p.poll() is None]
        if time.time() - printed > 20.0:
            printed = time.time()
            progress = progress_totals(out_root)
            print(json.dumps({
                "elapsed": round(time.time() - started, 1),
                "workers_alive": len(alive),
                "frames": progress["frames"],
                "deaths": progress["counts"]["death"],
                "bombs": progress["counts"]["bomb"],
                "hits": progress["counts"]["hit"],
                "pixel_gib": round(progress["pixel_bytes"] / (1024 ** 3), 3),
                "fps": round(progress["frames"] / max(1e-9, time.time() - started), 1),
            }, ensure_ascii=False), flush=True)
        if not alive:
            break
        time.sleep(2.0)
    for index, proc, log in procs:
        exit_codes[index] = proc.wait()
        log.close()
    report = scan(out_root, planes=True)
    report["exit_codes"] = exit_codes
    report["wall_seconds"] = round(time.time() - started, 1)
    (out_root / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# stats / index
# ---------------------------------------------------------------------------
def iter_attempts(out_dir: str | Path) -> Iterable[tuple[Path, Path]]:
    """``(shard_root, attempt_dir)`` for every attempt that holds frames."""

    out_dir = Path(out_dir)
    root = out_dir / "shards"
    if not root.is_dir():
        return
    for shard_root in sorted(p for p in root.iterdir() if p.is_dir()):
        for attempt in sorted(p for p in shard_root.glob("attempt-*") if p.is_dir()):
            if shard_frame_count(attempt) > 0:
                yield shard_root, attempt


def scan(out_dir: str | Path, *, planes: bool = True) -> dict[str, Any]:
    """Aggregate every attempt's label rows into one audit dict."""

    counts = {name: 0 for name in ("death", "bomb", "hit", "dying_started",
                                   "effect_spawned", "game_over_frames",
                                   "game_over_hud_frames", "hud_missing",
                                   "events_missing", "state_missing",
                                   "score_up", "score_same", "score_down")}
    frames = 0
    frames_death_negative = 0
    pixel_bytes = 0
    pixel_bytes_by_plane = {name: 0 for name in frame_bytes()}
    sources: dict[str, int] = {}
    stages: dict[str, int] = {}
    difficulties: dict[str, int] = {}
    shards: list[dict[str, Any]] = []
    death_events = 0
    bomb_events = 0
    hit_events = 0
    engines: dict[str, int] = {}
    for shard_root, attempt in iter_attempts(out_dir):
        rows = read_labels(attempt)
        if not rows:
            continue
        try:
            meta = json.loads((attempt / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        flags_key = "ev"
        local = {name: 0 for name in counts}
        prev_death = False
        prev_bomb = False
        prev_hit = False
        for row in rows:
            flags = row.get(flags_key)
            if flags is None:
                local["events_missing"] += 1
                death = bomb = hit = dying = False
            else:
                death = bool(int(flags) & EVENT_BIT_DEATH)
                bomb = bool(int(flags) & EVENT_BIT_BOMB)
                hit = bool(int(flags) & EVENT_BIT_HIT)
                dying = bool(int(flags) & EVENT_BIT_DYING)
                local["dying_started"] += int(dying)
                local["effect_spawned"] += int(bool(int(flags) & EVENT_BIT_EFFECT))
            local["death"] += int(death)
            local["bomb"] += int(bomb)
            local["hit"] += int(hit)
            death_events += int(death and not prev_death)
            bomb_events += int(bomb and not prev_bomb)
            hit_events += int(hit and not prev_hit)
            prev_death, prev_bomb, prev_hit = death, bomb, hit
            local["game_over_frames"] += int(row.get("go") or 0)
            local["game_over_hud_frames"] += int(row.get("hgo") or 0)
            local["hud_missing"] += int(row.get("hgo") is None)
            if row.get("ps") is None:
                local["state_missing"] += 1
        # score deltas, computed from the stored rows (not from meta counters, so
        # a killed worker's tail is still classified)
        for prev, cur in zip(rows, rows[1:]):
            if prev.get("sc") is None or cur.get("sc") is None:
                continue
            delta = int(cur["sc"]) - int(prev["sc"])
            local["score_up" if delta > 0 else ("score_down" if delta < 0 else "score_same")] += 1
        for name, value in local.items():
            counts[name] += value
        frames += len(rows)
        frames_death_negative += len(rows) - local["death"]
        spec = meta.get("spec") or {}
        sources[str(spec.get("source"))] = sources.get(str(spec.get("source")), 0) + len(rows)
        stages[str(spec.get("stage"))] = stages.get(str(spec.get("stage")), 0) + len(rows)
        difficulties[str(spec.get("difficulty"))] = (
            difficulties.get(str(spec.get("difficulty")), 0) + len(rows)
        )
        sha = meta.get("engine_sha256")
        if sha:
            engines[sha] = engines.get(sha, 0) + 1
        sizes = {}
        for name, size in frame_bytes().items():
            path = plane_path(attempt, name)
            if path.is_file():
                sizes[name] = path.stat().st_size
                pixel_bytes_by_plane[name] += sizes[name]
                pixel_bytes += sizes[name]
        shards.append(
            {
                "shard": shard_root.name,
                "attempt": attempt.name,
                "frames": len(rows),
                "complete": bool(meta.get("complete")),
                "end_reason": meta.get("end_reason"),
                "counts": {k: local[k] for k in ("death", "bomb", "hit", "dying_started")},
                "bytes": sizes,
            }
        )
    report = {
        "schema": "v400-p2det-stats-1",
        "out_dir": str(out_dir),
        "frames_total": frames,
        "shards": len(shards),
        "counts": counts,
        "events": {
            "death": death_events,
            "bomb": bomb_events,
            "hit": hit_events,
        },
        "sources": dict(sorted(sources.items())),
        "stages": dict(sorted(stages.items())),
        "difficulties": dict(sorted(difficulties.items())),
        "engine_sha256": engines,
        "pixel_bytes": pixel_bytes,
        "pixel_gib": round(pixel_bytes / (1024 ** 3), 3),
        "pixel_bytes_by_plane": pixel_bytes_by_plane,
        "frame_bytes": sum(frame_bytes().values()),
        "shard_rows": shards,
        "g2_gap": g2_gap(frames, {"death": counts["death"], "bomb": counts["bomb"],
                                  "hit": counts["hit"]}),
    }
    if planes:
        report["plane_layout"] = {name: list(shape) for name, shape in plane_shapes().items()}
    return report


def g2_gap(frames: int, positives: dict[str, int]) -> dict[str, Any]:
    """How much data the G2 gate still needs (not a detector score).

    Rule of three: zero failures in ``n`` trials gives a one-sided 95 % lower
    bound of ``1 - 3/n``, so ``>= 99 %`` needs ``n >= 299`` positive events and
    ``>= 99.5 %`` needs ``n >= 599``.
    """

    need = {"0.99": 299, "0.995": 599}
    return {
        "frames": frames,
        "positives": positives,
        "needed_99pct_lower_bound": need["0.99"],
        "needed_99_5pct_lower_bound": need["0.995"],
        "death_shortfall": max(0, need["0.99"] - positives.get("death", 0)),
        "death_gate_met": positives.get("death", 0) >= need["0.99"],
    }


def index_command(args) -> int:
    entries = []
    for shard_root, attempt in iter_attempts(args.out_dir):
        try:
            meta = json.loads((attempt / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        spec = meta.get("spec") or {}
        frames = shard_frame_count(attempt)
        if frames == 0:
            continue
        entries.append(
            {
                "shard": shard_root.name,
                "attempt": attempt.name,
                "dir": str(attempt),
                "frames": frames,
                "source": spec.get("source"),
                "stage": spec.get("stage"),
                "difficulty": spec.get("difficulty"),
                "seed": spec.get("seed"),
                "complete": bool(meta.get("complete")),
                "engine_sha256": meta.get("engine_sha256"),
                "label_version": meta.get("label_version"),
                "end_reason": meta.get("end_reason"),
            }
        )
    index = {
        "schema": "v400-p2det-index-1",
        "out_dir": str(Path(args.out_dir)),
        "frames_total": sum(entry["frames"] for entry in entries),
        "shards": entries,
        "plane_shapes": {name: list(shape) for name, shape in plane_shapes().items()},
        "label_version": LABEL_VERSION,
    }
    path = Path(args.out_dir) / "index.json"
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"index": str(path), "frames": index["frames_total"],
                      "shards": len(entries)}, ensure_ascii=False))
    return 0


def stats_command(args) -> int:
    report = scan(args.out_dir, planes=True)
    out = Path(args.out) if args.out else Path(args.out_dir) / "stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    brief = {
        key: report[key]
        for key in ("frames_total", "shards", "counts", "events", "sources",
                    "stages", "difficulties", "pixel_gib", "g2_gap", "engine_sha256")
    }
    brief["stats"] = str(out)
    print(json.dumps(brief, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="create per-instance data dirs")
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--instances", type=int, required=True)
    prepare.add_argument("--assets", default="/root/v400-data/game")
    prepare.add_argument("--template", default="/root/v400-data/p2/game")

    worker = sub.add_parser("worker", help="one process, its own shard list")
    worker.add_argument("--engine", required=True)
    worker.add_argument("--data-root", required=True)
    worker.add_argument("--out-dir", required=True)
    worker.add_argument("--plan", required=True)
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--frames", type=int, required=True)
    worker.add_argument("--max-seconds", type=float, default=0.0)

    run = sub.add_parser("run", help="many parallel workers")
    run.add_argument("--engine", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--instances", type=int, default=16)
    run.add_argument("--frames-total", type=int, default=300_000)
    run.add_argument("--frames-per-shard", type=int, default=3200)
    run.add_argument("--max-frames", type=int, default=None,
                     help="alias of --frames-per-shard; the per-game frame cap that "
                          "bounds how many lives a shard can burn")
    run.add_argument("--seed-base", type=int, default=1000)
    run.add_argument("--hold", type=int, default=6)
    run.add_argument("--bomb-prob", type=float, default=0.02)
    run.add_argument("--warmup", type=int, default=900)
    run.add_argument("--lp-threads", type=int, default=4)
    run.add_argument("--policy-plan", default=",".join(POLICY_PLAN),
                     help="comma separated policy cycle, e.g. random,still,dive")
    run.add_argument("--stage-plan", default=None,
                     help="override the stage cycle for a top-up run, e.g. '3:3' or "
                          "'3:3,3:1'; default keeps the P2 STAGE_PLAN")
    run.add_argument("--max-seconds", type=float, default=0.0)
    run.add_argument("--assets", default="/root/v400-data/game")
    run.add_argument("--template", default="/root/v400-data/p2/game")
    run.add_argument("--hud", action="store_true",
                     help="also poll the read-only HUD command every frame "
                          "(rl_score build); adds hli/hbo/hgo/hage/hstale fields")

    stats = sub.add_parser("stats", help="frame/positive/byte audit")
    stats.add_argument("--out-dir", required=True)
    stats.add_argument("--out", default=None)

    index = sub.add_parser("index", help="write index.json for training")
    index.add_argument("--out-dir", required=True)

    stats.set_defaults(func=stats_command)
    index.set_defaults(func=index_command)
    run.set_defaults(func=run_command)
    worker.set_defaults(func=worker_command)
    prepare.set_defaults(
        func=lambda a: (
            print(json.dumps(prepare_instances(
                data_root=a.data_root, instances=a.instances,
                assets=a.assets, template=a.template), ensure_ascii=False)),
            0,
        )[1]
    )
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
