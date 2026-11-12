"""Replay D0's recorded per-frame masks and re-render the native playfield.

Why a replay and not a re-collection: EXP-P1-014 already proved that an env-role
engine fed the same ``RESET`` and the same per-frame button masks produces the
same 25-field ``HASH`` at every decision boundary (6740/6740 checkpoints equal on
seed 1).  D0 therefore *does* carry everything a byte-exact re-render needs:

* the StageStart -- ``stage/difficulty/team/seed`` in ``shard.json``, identical
  to ``work/seedN/channel_b.json``'s ``reset_command`` (``RESET 0 1 0 N 0``);
* the per-frame masks -- ``labels.jsonl``'s ``step_frames``, four per decision,
  which ``v400_d0_dual.load_labels`` already validates as the mask sequence.

What D0 does *not* carry is the native 384x448 picture: it stored obs-v1's
64x64 blocks only (``obs/play64.npy``).  So this module re-runs channel B at
native resolution and rewrites the pixels, keeping D0's labels untouched.

The gate is the same one channel B used: every decision boundary's ``HASH``
must equal ``labels.jsonl``'s ``hash_pre``/``hash_post``.  A single mismatch
aborts the seed -- no partial shard is published.

Per decision the collector also records (from engine *state*, never pixels):

* ``PLAYER`` position and hurtbox -> the label window's anchor;
* ``HAZARDS`` -> the P2 96x112 grid, cropped to the 24x24 window around the ship;
* the future-8 danger map, from the distance fields of the eight frames that
  follow the decision instant (they are exactly the four STEP renders of this
  decision and of the next one, so the rolling buffer is nine frames deep);
* the P2 still-ship verdict ``evaluate_future_danger`` at horizons 4 and 8.

One ``HAZARDS`` call per *frame* is enough: the last STEP render of decision
``d`` is the newest render of decision ``d+1`` (frame ``(d+1)*4 - 1``), so the
current-frame grid is carried over instead of being queried twice.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from touhou_rl.v400.gym.protocol import (
    HazardFrame,
    PlayerSnapshot,
    parse_hazards_reply,
    parse_player_reply,
)
from touhou_rl.v400.gym.shm import PIXELS_OFFSET, FLAG_PIXELS_VALID, SharedBlock, segment_path
from touhou_rl.v400.labels.danger import evaluate_future_danger
from touhou_rl.v400.labels.player import player_label
from touhou_rl.v400.labels.runner import ControlClient
from touhou_rl.v400.student.probe.labels import (
    DANGER_HORIZON,
    grid_arrays,
    min_distance,
    snap_anchor,
    window_label,
)
from touhou_rl.v400.student.probe.native import (
    PLAYFIELD_HEIGHT,
    PLAYFIELD_WIDTH,
    gray_playfield,
)

#: Frames per teacher decision; D0 wrote exactly this many masks per row.
FRAMES_PER_DECISION = 4
#: Big-endian readback, RGBA8, ``rlharness.cpp`` PublishPixels.
PIXEL_BYTES = 640 * 480 * 4
#: Files the engine may write; the two archives stay read-only symlinks.
LINKED_FILES = ("th08.dat", "thbgm.dat")
COPIED_FILES = ("th08.cfg", "score.dat")
#: How many future frames the danger map consumes (task book: 8).
DANGER_FRAMES = DANGER_HORIZON


class CollectError(RuntimeError):
    """A precondition, a protocol reply, or the replay gate failed."""


class HashMismatch(CollectError):
    """The replayed trajectory is not D0's trajectory."""


# ---------------------------------------------------------------------------
# replay plan (read D0, never write it)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReplayRow:
    """One decision's replay record, mirroring the D0 label invariants."""

    decision: int
    step_frames: tuple[int, ...]
    hash_pre: str
    hash_post: str
    frame_pre: int
    frame_post: int


def _hash_field(payload: Any, name: str, where: str,
                expected_frame: int | None = None) -> str:
    """Accept both D0 hash spellings: a bare digest, or ``{hash, frames}``.

    The *assembled* shard stores ``"hash_pre": "7f04c7361749ed9f"``; channel A's
    raw ``work/<seed>/labels.jsonl`` stores ``{"hash": ..., "frames": ...}``.
    Both are the same engine checkpoint; only the frame counter is implicit in
    the shard form, where it is ``decision * FRAMES_PER_DECISION`` by construction.
    """

    if isinstance(payload, dict):
        value = str(payload.get("hash", ""))
        if "frames" in payload and expected_frame is not None:
            if int(payload["frames"]) != int(expected_frame):
                raise CollectError(
                    f"{where}: {name} frame {payload['frames']} != {expected_frame}")
    elif isinstance(payload, str):
        value = payload
    else:
        raise CollectError(f"{where}: {name} is not a hash: {payload!r}")
    if len(value) != 16 or any(ch not in "0123456789abcdef" for ch in value):
        raise CollectError(f"{where}: {name} is not an engine hash: {value!r}")
    return value


def load_replay_rows(path: str | Path, *, limit: int | None = None) -> list[ReplayRow]:
    """Read ``labels.jsonl`` and prove it is a replayable mask sequence."""

    rows: list[ReplayRow] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("kind") != "decision":
                continue
            where = f"{path}:{line_number}"
            index = len(rows)
            if int(payload.get("decision", -1)) != index:
                raise CollectError(f"{where}: decision {payload.get('decision')!r} != {index}")
            masks = [int(value) for value in payload.get("step_frames") or []]
            if len(masks) != FRAMES_PER_DECISION:
                raise CollectError(
                    f"{where}: {len(masks)} per-frame masks, need {FRAMES_PER_DECISION}")
            frame_pre = index * FRAMES_PER_DECISION
            frame_post = (index + 1) * FRAMES_PER_DECISION
            pre_hash = _hash_field(payload.get("hash_pre"), "hash_pre", where, frame_pre)
            post_hash = _hash_field(payload.get("hash_post"), "hash_post", where, frame_post)
            rows.append(ReplayRow(
                decision=index,
                step_frames=tuple(masks),
                hash_pre=pre_hash,
                hash_post=post_hash,
                frame_pre=frame_pre,
                frame_post=frame_post,
            ))
            if limit is not None and len(rows) >= int(limit):
                break
    if not rows:
        raise CollectError(f"{path}: no replayable decision rows")
    return rows


def read_stage_start(shard_dir: str | Path) -> dict[str, Any]:
    """StageStart + engine identity from the D0 shard, or fail closed."""

    shard = json.loads((Path(shard_dir) / "shard.json").read_text(encoding="utf-8"))
    start = {
        "stage": int(shard["stage"]),
        "difficulty": int(shard["difficulty"]),
        "team": int(shard["team"]),
        "seed": int(shard["seed"]),
    }
    return {
        "stage_start": start,
        "reset_command": (f"RESET {start['stage']} {start['difficulty']} "
                          f"{start['team']} {start['seed']} 0"),
        "shard": str(shard_dir),
        "decisions": int(shard.get("decisions", shard.get("decision_count", 0))),
        "obs_spec_version": shard.get("obs_spec_version"),
        "stage_frames": shard.get("stage_frames"),
    }


def prepare_data_dir(template: Path, target: Path) -> dict[str, Any]:
    """A fresh data dir: read-only archives symlinked, writable files copied."""

    if not template.is_dir():
        raise CollectError(f"data template {template} is not a directory")
    if target.exists():
        return {"data_dir": str(target), "reused": True}
    target.mkdir(parents=True)
    for name in LINKED_FILES:
        source = template / name
        if not source.exists():
            raise CollectError(f"data template lacks {source}")
        (target / name).symlink_to(source)
    copied = []
    for name in COPIED_FILES:
        source = template / name
        if source.exists():
            shutil.copy2(source, target / name)
            copied.append(name)
    return {"data_dir": str(target), "reused": False,
            "linked": list(LINKED_FILES), "copied": copied}


# ---------------------------------------------------------------------------
# the replay itself
# ---------------------------------------------------------------------------
@dataclass
class _Pending:
    """One decision waiting for its eight future frames."""

    decision: int
    instant_frame: int
    observed_frame: int
    anchor_x: int
    anchor_y: int
    player: dict[str, Any]
    occupied: np.ndarray
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Frame:
    """One rendered frame's hazard state."""

    frame: int
    hazards: Any
    occupied: np.ndarray
    distance: np.ndarray


def _tokens(reply: str) -> dict[str, str]:
    return {part.split("=", 1)[0]: part.split("=", 1)[1]
            for part in reply.split()[1:] if "=" in part}


def _hash_reply(client: ControlClient) -> tuple[str, int]:
    fields = _tokens(client.request("HASH"))
    value = str(fields.get("hash", ""))
    if len(value) != 16:
        raise CollectError(f"HASH reply is not an engine hash: {fields!r}")
    if "frames" not in fields:
        raise CollectError(f"HASH reply has no counter: {fields!r}")
    return value, int(fields["frames"])


def _hazards(client: ControlClient) -> HazardFrame:
    """``HAZARDS`` (all slots) through the frozen parser the labels use."""

    return parse_hazards_reply(client.request("HAZARDS", timeout=120.0))


def _player(client: ControlClient) -> PlayerSnapshot:
    return parse_player_reply(client.request("PLAYER"))


def _read_native(block: SharedBlock, where: str) -> tuple[np.ndarray, dict[str, int]]:
    header = block.header()
    if not int(header.get("flags", 0)) & FLAG_PIXELS_VALID:
        raise CollectError(f"{where}: shared block has no published pixels")
    if int(header.get("width", 0)) != 640 or int(header.get("height", 0)) != 480:
        raise CollectError(
            f"{where}: readback is {header.get('width')}x{header.get('height')}")
    native = gray_playfield(block.pixels())
    if native.shape != (PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH):
        raise CollectError(f"{where}: native field {native.shape} is not 384x448")
    if not native.any():
        raise CollectError(f"{where}: readback is all zero; refusing a blank frame")
    return native, header


def collect_seed(
    *,
    seed: int,
    shard_dir: Path,
    binary: Path,
    data_template: Path,
    data_dir: Path,
    out_dir: Path,
    expected_sha256: str | None = None,
    limit: int | None = None,
    timeout: float = 300.0,
    log=print,
) -> dict[str, Any]:
    """Replay one D0 seed at native resolution and write ``s{seed}.npy``+``.npz``."""

    from touhou_rl.v400.teacher.engine import sha256_file

    started = time.monotonic()
    if expected_sha256 is not None:
        actual = sha256_file(binary)
        if actual != expected_sha256:
            raise CollectError(
                f"{binary}: sha256 {actual} != expected {expected_sha256}")
    stage = read_stage_start(shard_dir)
    rows = load_replay_rows(Path(shard_dir) / "labels.jsonl", limit=limit)
    log(f"[probe] seed {seed}: {len(rows)} decisions, {stage['reset_command']}")

    data = prepare_data_dir(Path(data_template), Path(data_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    native_path = out_dir / f"native_s{seed}.npy"
    shm_name = f"v400p3diag-{os.getpid()}-{int(time.time() * 1000) % 10_000_000}"

    count = len(rows)
    native = np.lib.format.open_memmap(
        native_path, mode="w+", dtype=np.uint8,
        shape=(count, PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH))

    cells = np.zeros((count, 24, 24), dtype=bool)
    valid = np.zeros((count, 24, 24), dtype=bool)
    danger = np.zeros((count, 24, 24), dtype=bool)
    anchors = np.zeros((count, 2), dtype=np.int16)
    players = np.zeros((count, 2), dtype=np.float32)
    hurt = np.zeros((count, 4), dtype=np.float32)
    would_die = np.full((count, 2), -1, dtype=np.int8)   # horizons 4, 8
    danger_frames = np.zeros(count, dtype=np.int8)
    instant_frames = np.zeros(count, dtype=np.int64)
    observed_frames = np.zeros(count, dtype=np.int64)
    hashes_pre: list[str] = []
    hashes_post: list[str] = []
    mismatches: list[dict[str, Any]] = []

    pending: list[_Pending] = []
    window: list[_Frame] = []
    client = ControlClient(binary, data["data_dir"], role="env", timeout=timeout,
                           stderr_path=out_dir / f"engine_s{seed}.log")
    block: SharedBlock | None = None
    grid_ms = 0.0
    pixel_ms = 0.0
    filled = 0
    try:
        client.start()
        hello = _tokens(client.request("HELLO"))
        if hello.get("role") != "env":
            raise CollectError(f"HELLO did not confirm the env role: {hello!r}")
        client.request(f"SHM /{shm_name}")
        readback = _tokens(client.request("READBACK 1"))
        if readback.get("readback") != "1":
            raise CollectError(f"READBACK 1 refused: {readback!r}")
        block = SharedBlock(shm_name)
        if block.size != PIXELS_OFFSET + PIXEL_BYTES:
            raise CollectError(
                f"shared block is {block.size} B, expected {PIXELS_OFFSET + PIXEL_BYTES}")
        reset = _tokens(client.request(stage["reset_command"], timeout=timeout))
        if reset.get("ready") != "1":
            raise CollectError(f"RESET did not become ready: {reset!r}")
        frame_now = int(_tokens(client.request("FRAME")).get("frame", -1))
        if frame_now != 0:
            raise CollectError(f"RESET left the frame counter at {frame_now}")

        # Frame -1 (the render RESET left behind) is decision 0's newest picture.
        seeded = time.perf_counter()
        hazards = _hazards(client)
        occupied, distance = grid_arrays(hazards)
        grid_ms += (time.perf_counter() - seeded) * 1000.0
        window.append(_Frame(frame=-1, hazards=hazards,
                             occupied=occupied, distance=distance))

        def finalize(upto_frame: int) -> None:
            nonlocal filled
            while pending and pending[0].instant_frame + DANGER_FRAMES <= upto_frame:
                record = pending.pop(0)
                future = [entry for entry in window
                          if record.instant_frame < entry.frame
                          <= record.instant_frame + DANGER_FRAMES]
                fields = min_distance(entry.distance for entry in future) \
                    if future else None
                verdict = evaluate_future_danger(
                    record.player, [entry.hazards for entry in future], horizons=(4, 8))
                label = window_label(
                    record.occupied, fields, record.player.position[0],
                    record.player.position[1],
                    danger_frames=len(future),
                    would_die_8=verdict.per_horizon.get(8),
                )
                index = record.decision
                native[index] = record.summary.pop("native")
                cells[index] = label.cells
                valid[index] = label.valid
                danger[index] = label.danger
                anchors[index] = (label.anchor_y // 4, label.anchor_x // 4)
                players[index] = (label.player_x, label.player_y)
                hurt_min = record.player.hurtbox_min
                hurt_max = record.player.hurtbox_max
                hurt[index] = (hurt_min[0], hurt_min[1], hurt_max[0], hurt_max[1])
                for slot, horizon in enumerate((4, 8)):
                    value = verdict.per_horizon.get(horizon)
                    would_die[index, slot] = -1 if value is None else int(bool(value))
                danger_frames[index] = len(future)
                instant_frames[index] = record.instant_frame
                observed_frames[index] = record.observed_frame
                filled += 1

        for row in rows:
            index = row.decision
            frame_now = int(_tokens(client.request("FRAME")).get("frame", -1))
            if frame_now != row.frame_pre:
                raise CollectError(
                    f"decision {index}: engine frame {frame_now} != {row.frame_pre}")
            pre_hash, pre_frames = _hash_reply(client)
            hashes_pre.append(pre_hash)
            if pre_hash != row.hash_pre or pre_frames != row.frame_pre:
                mismatches.append({
                    "decision": index, "boundary": "pre", "frame": row.frame_pre,
                    "d0": row.hash_pre, "replay": pre_hash, "replay_frames": pre_frames,
                })
                break

            started_px = time.perf_counter()
            native_pixels, header = _read_native(block, f"decision {index}")
            pixel_ms += (time.perf_counter() - started_px) * 1000.0
            player = _player(client)
            current = window[-1]
            anchor_x, anchor_y = snap_anchor(player.position[0], player.position[1])
            pending.append(_Pending(
                decision=index,
                instant_frame=current.frame,
                observed_frame=int(header["frame"]),
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                player=player,
                occupied=current.occupied,
                summary={"native": native_pixels},
            ))

            for offset in range(FRAMES_PER_DECISION):
                mask = int(row.step_frames[offset])
                reply = _tokens(client.request(f"STEP {mask} 1"))
                if int(reply.get("advanced", 0)) != 1:
                    raise CollectError(
                        f"decision {index} frame {offset}: advanced="
                        f"{reply.get('advanced')} on STEP {mask} 1")
                started_grid = time.perf_counter()
                hazards = _hazards(client)
                occupied, distance = grid_arrays(hazards)
                grid_ms += (time.perf_counter() - started_grid) * 1000.0
                window.append(_Frame(frame=row.frame_pre + offset, hazards=hazards,
                                     occupied=occupied, distance=distance))

            post_hash, post_frames = _hash_reply(client)
            hashes_post.append(post_hash)
            if post_hash != row.hash_post or post_frames != row.frame_post:
                mismatches.append({
                    "decision": index, "boundary": "post", "frame": row.frame_post,
                    "d0": row.hash_post, "replay": post_hash,
                    "replay_frames": post_frames,
                })
                break
            finalize(window[-1].frame)
            # Nine frames is all the danger map needs; drop the rest so a long
            # replay does not hold every hazard snapshot in memory.
            while len(window) > DANGER_FRAMES + 1:
                window.pop(0)

        # Tail: decisions whose eight future frames do not exist yet.
        finalize(10 ** 18 if not mismatches else -1)
    finally:
        if block is not None:
            try:
                block.close()
            except Exception:  # pragma: no cover - close races with QUIT
                pass
        try:
            client.request("QUIT", timeout=20.0)
        except Exception:  # pragma: no cover - QUIT may lose the race
            pass
        client.close()
        try:
            segment_path(shm_name).unlink()
        except (OSError, ValueError):
            pass

    native.flush()
    del native
    wall = time.monotonic() - started
    report: dict[str, Any] = {
        "kind": "v400-p3diag-collect-1",
        "seed": int(seed),
        "shard": str(shard_dir),
        "stage_start": stage,
        "engine_binary": str(binary),
        "engine_sha256": expected_sha256,
        "decisions_requested": count,
        "decisions_filled": filled,
        "frames_replayed": len(hashes_post),
        "hash_mismatches": mismatches[:4],
        "hash_mismatch_count": len(mismatches),
        "status": "ok" if not mismatches else "HASH_MISMATCH",
        "wall_s": round(wall, 3),
        "frames_per_s": round(len(hashes_post) * FRAMES_PER_DECISION / wall, 3) if wall else 0.0,
        "pixel_ms_total": round(pixel_ms, 3),
        "grid_ms_total": round(grid_ms, 3),
        "data_dir": data,
        "native_path": str(native_path),
        "native_shape": [count, PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH],
        "native_bytes": int(native_path.stat().st_size) if native_path.exists() else 0,
        "requests": int(client.requests),
    }
    if mismatches:
        # Fail closed: no shard is published for a seed that is not D0.
        for path in (native_path,):
            try:
                path.unlink()
            except OSError:
                pass
        report["published"] = False
        raise HashMismatch(json.dumps(report, ensure_ascii=False))

    labels_path = out_dir / f"labels_s{seed}.npz"
    np.savez_compressed(
        labels_path,
        cells=np.packbits(cells.reshape(count, -1), axis=1),
        valid=np.packbits(valid.reshape(count, -1), axis=1),
        danger=np.packbits(danger.reshape(count, -1), axis=1),
        anchors=anchors,
        players=players,
        hurt=hurt,
        would_die=would_die,
        danger_frames=danger_frames,
        instant_frames=instant_frames,
        observed_frames=observed_frames,
        hashes_pre=np.asarray(hashes_pre, dtype="U16"),
        hashes_post=np.asarray(hashes_post, dtype="U16"),
        stage_start=np.asarray([stage["stage_start"][key] for key in
                                ("stage", "difficulty", "team", "seed")], dtype=np.int64),
    )
    from touhou_rl.v400.teacher.engine import sha256_file

    report.update({
        "published": True,
        "labels_path": str(labels_path),
        "labels_bytes": int(labels_path.stat().st_size),
        "native_sha256": sha256_file(native_path),
        "labels_sha256": sha256_file(labels_path),
        "positives_total": int(cells[valid].sum()),
        "scored_cells_total": int(valid.sum()),
        "danger_cells_total": int(danger[valid].sum()),
        "full_windows": int(valid.reshape(count, -1).all(axis=1).sum()),
        "would_die_8": int((would_die[:, 1] == 1).sum()),
        "would_die_8_known": int((would_die[:, 1] >= 0).sum()),
    })
    (out_dir / f"collect_s{seed}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


__all__ = [
    "CollectError",
    "DANGER_FRAMES",
    "FRAMES_PER_DECISION",
    "HashMismatch",
    "ReplayRow",
    "collect_seed",
    "load_replay_rows",
    "prepare_data_dir",
    "read_stage_start",
]
