#!/usr/bin/env python3
"""P2 labels S4: collect real frames with labels+pixels, overlay 50, audit.

Three subcommands, split by where their dependencies live:

``collect`` (WSL guest, where the engine and its /dev/shm live)
    Drives ``build/rl_hazard/th08-modern`` in ``env`` role with ``READBACK 1``
    and ``SHM`` attached, steps real gameplay (stage 1 Normal + stage 4 Lunatic,
    ``still`` and ``random`` policies, fixed seeds), and writes one JSONL line
    per frame with **state-derived labels** plus a lossless PNG of the same
    frame.  The pixel bytes never enter a label computation.

``overlay`` (Windows side, Pillow)
    Picks 50 frames, draws player point / hurtbox / occupancy grid / HUD values
    on top of their own pixels, writes 50 PNGs + one contact sheet, and runs the
    coordinate checks (playfield origin edge measurement, focus-hitbox red-dot
    diagnostic, occupancy-vs-bright-pixel alignment).

``audit`` (Windows side, numpy)
    Builds the P2-WP2 audit table skeleton from the JSONL, and optionally
    measures the HUD redraw lag from the pixel bands.

Engine facts used (all from the fixed ``rl_hazard`` build, SHA 8053dfaa…):
``HAZARDS`` + ``PLAYER`` + ``STATE`` per frame, ``READBACK 1`` publishing
640x480 RGBA into ``/dev/shm/<name>``, playfield viewport (32, 16, 384, 448)
(``src/Background.cpp:740-743``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Buttons, from src/Global.hpp TouhouButton.
BUTTON_SHOOT = 1 << 0
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


class EngineError(RuntimeError):
    """The engine replied ``ERR`` or the control pipe died."""


class ControlClient:
    """Bare line-protocol client for the env role (keeps DISPLAY for rendering)."""

    def __init__(
        self,
        binary: str,
        data_dir: str,
        *,
        role: str = "env",
        timeout: float = 300.0,
        stderr_path: Path | None = None,
    ) -> None:
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
        env.update(
            SDL_AUDIODRIVER="dummy",
            LIBGL_ALWAYS_SOFTWARE="1",
            GALLIUM_DRIVER="llvmpipe",
            LP_NUM_THREADS="1",
        )
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


def parse_reply(reply: str) -> dict[str, str]:
    parts = reply.split()
    out = {"_verb": parts[0] if parts else ""}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Policy:
    """Seeded action source; same definition as EXP-P1-004's collector."""

    def __init__(self, name: str, rng: random.Random, hold: int = 6) -> None:
        if name not in ("still", "random"):
            raise SystemExit(f"unknown policy {name!r}")
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


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
def collect(args) -> int:
    import numpy as np

    from touhou_rl.v400.gym.protocol import parse_hazards_reply, parse_player_reply
    from touhou_rl.v400.gym.shm import SharedBlock
    from touhou_rl.v400.labels import (
        LABEL_VERSION,
        PLAYER_STATE_ALIVE,
        PLAYER_STATE_DYING,
        SCREEN_ORIGIN,
        decode_state,
        encode_occupancy,
        evaluate_future_danger,
        event_labels,
        hazard_occupancy,
        hud_label,
        player_label,
        state_from_reply,
    )
    from touhou_rl.v400.labels.png import write_rgba_png

    engine = Path(args.engine)
    if not engine.is_file():
        raise SystemExit(f"engine binary not found: {engine}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"")
    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    engine_sha = sha256_file(engine)
    meta: dict = {
        "schema": "v400-p2-labels-collect-1",
        "engine": str(engine),
        "engine_sha256": engine_sha,
        "data_dir": str(Path(args.data_dir).resolve()),
        "role": args.role,
        "label_version": LABEL_VERSION,
        "screen_origin": list(SCREEN_ORIGIN),
        "playfield": [384, 448],
        "grid": [96, 112, 4],
        "horizon": 16,
        "geometry_model": "engine-phase",
        "occupancy_model": "spec",
        "runs_requested": list(args.runs),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs_meta": [],
    }

    frames_written = 0
    started = time.monotonic()
    for run_index, spec in enumerate(args.runs):
        bits = spec.split(":")
        if len(bits) not in (4, 5, 6):
            raise SystemExit(
                f"bad --runs entry {spec!r} "
                "(stage:difficulty:policy:frames[:policySeed[:resetSeed]])"
            )
        stage, difficulty, policy, frames = (
            int(bits[0]),
            int(bits[1]),
            bits[2],
            int(bits[3]),
        )
        policy_seed = int(bits[4]) if len(bits) >= 5 else args.seed * 1000 + run_index
        reset_seed = int(bits[5]) if len(bits) == 6 else args.seed
        actions = Policy(policy, random.Random(policy_seed), args.hold)
        shm_name = f"v400p2-{os.getpid()}-{run_index}"
        stderr_path = out.with_suffix(f".engine.stderr.run{run_index}.log")
        client = ControlClient(
            engine, args.data_dir, role=args.role, timeout=args.timeout,
            stderr_path=stderr_path,
        )
        run_started = time.monotonic()
        raw: list[dict] = []
        reset_reply = None
        hello_reply = None
        error = None
        readback_reply = None
        shm_reply = None
        game_over_frame = None
        frame_mismatches = 0
        reset_seconds = None
        final_readback = None
        warmup_steps = 0
        shm = None
        try:
            client.start()
            hello_reply = client.request("HELLO")
            hello = parse_reply(hello_reply)
            if hello.get("role") != args.role:
                raise EngineError(
                    f"engine role is {hello.get('role')!r}, expected {args.role!r}: "
                    f"{hello_reply!r}"
                )
            shm_reply = client.request(f"SHM /{shm_name}")
            readback_reply = client.request("READBACK 1")
            if "readback=1" not in readback_reply:
                raise EngineError(f"READBACK 1 refused: {readback_reply!r}")
            shm = SharedBlock(shm_name)
            reset_started = time.monotonic()
            reset_reply = client.request(
                f"RESET {stage} {difficulty} 0 {reset_seed} 0", timeout=240.0
            )
            reset_seconds = time.monotonic() - reset_started
            if "ready=0" in reset_reply:
                raise EngineError(f"RESET did not become ready: {reset_reply}")

            def step_and_read(index: int, mask: int, record: bool) -> dict | None:
                nonlocal frame_mismatches, game_over_frame
                stepped = parse_reply(client.request(f"STEP {mask} 1"))
                if int(stepped.get("advanced", 0)) != 1:
                    # The chain asked to exit on this frame: no frame advanced,
                    # so there is nothing new to record.  Raised so the run ends
                    # and every already-collected frame is still labelled.
                    game_over_frame = index
                    raise EngineError(f"STEP did not advance: {stepped!r}")
                frame_number = int(stepped["frame"])
                # ``AfterDrawFrame`` runs before ``++g_frameCount``
                # (main.cpp:139-146), so the pixel buffer published during this
                # step carries ``frame - 1`` in the header while its *content*
                # is this frame's render.  Record both and keep the constant.
                header = shm.header()
                delta = frame_number - int(header["frame"])
                if delta != 1:
                    frame_mismatches += 1
                sent = time.monotonic()
                pixels = shm.pixels() if record else None
                pixel_seconds = time.monotonic() - sent
                if not record:
                    # The warm-up probe only needs to know when the stage's
                    # start-up invulnerability ends; reading PLAYER is enough
                    # and keeps the warm-up cheap.
                    player = parse_player_reply(client.request("PLAYER"))
                    return {"player_state": int(player.player_state)}
                hazards = parse_hazards_reply(client.request("HAZARDS"))
                player = parse_player_reply(client.request("PLAYER"))
                state = state_from_reply(client.request("STATE"))
                return {
                    "i": index,
                    "mask": mask,
                    "frame": frame_number,
                    "shm_frame": int(header["frame"]),
                    "pixel_stamp_delta": delta,
                    "game_over": stepped.get("game_over") == "1",
                    "hazards": hazards,
                    "player": player,
                    "state": state,
                    "pixels": pixels,
                    "pixel_seconds": pixel_seconds,
                }

            record_started = False
            warmup_steps = 0
            for index in range(-args.warmup, frames):
                mask = actions.mask(index if index >= 0 else 0)
                if not record_started:
                    probe = step_and_read(index, mask, False)
                    warmup_steps += 1
                    if probe is not None and probe["player_state"] == PLAYER_STATE_ALIVE:
                        record_started = True
                    continue
                item = step_and_read(len(raw), mask, True)
                assert item is not None
                raw.append(item)
                if len(raw) >= frames:
                    break
            final_readback = client.request("READBACK")
        except EngineError as exc:
            error = str(exc)
        finally:
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
            client.close()
            for path in Path("/dev/shm").glob(f"{shm_name}*"):
                try:
                    path.unlink()
                except OSError:
                    pass
        written = write_run_frames(
            out, frames_dir, run_index, stage, difficulty, policy, engine_sha, raw
        )
        frames_written += written
        run_meta = {
            "run": run_index,
            "stage": stage,
            "difficulty": difficulty,
            "policy": policy,
            "frames_requested": frames,
            "frames_recorded": len(raw),
            "frames_written": written,
            "warmup_steps": warmup_steps,
            "warmup_to_first_alive": record_started,
            "policy_seed": policy_seed,
            "reset_seed": reset_seed,
            "warmup": args.warmup,
            "hello": hello_reply,
            "shm": shm_reply,
            "readback": readback_reply,
            "readback_final": final_readback,
            "reset_reply": reset_reply,
            "reset_seconds": None if reset_seconds is None else round(reset_seconds, 3),
            "game_over_frame": game_over_frame,
            "shm_frame_mismatches": frame_mismatches,
            "pixel_stamp_delta_histogram": _histogram(
                [item["pixel_stamp_delta"] for item in raw]
            ),
            "pixel_read_seconds_total": round(
                sum(item["pixel_seconds"] for item in raw), 3
            ),
            "error": error,
            "wall_seconds": round(time.monotonic() - run_started, 3),
            "player_state_histogram": _histogram(
                [
                    int(item["player"].player_state)
                    for item in raw
                ]
            ),
            "death_proxy_events": sum(
                1
                for position, item in enumerate(raw[:-1])
                if int(item["player"].player_state) == PLAYER_STATE_ALIVE
                and int(raw[position + 1]["player"].player_state) == PLAYER_STATE_DYING
            ),
            "lives_vs_state_mismatches": sum(
                1
                for item in raw
                if int(item["player"].lives)
                != int(item["state"].get("lives_remaining") or 0)
            ),
            "max_bullets": max((item["hazards"].bullet_count for item in raw), default=0),
            "max_enemies": max((item["hazards"].enemy_count for item in raw), default=0),
            "max_lasers": max((item["hazards"].laser_count for item in raw), default=0),
        }
        meta["runs_meta"].append(run_meta)
        print(json.dumps(run_meta, ensure_ascii=False), flush=True)

    meta["frames_written"] = frames_written
    meta["wall_seconds"] = round(time.monotonic() - started, 3)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "frames_written": frames_written,
                "runs": len(meta["runs_meta"]),
                "wall_seconds": meta["wall_seconds"],
                "engine_sha256": engine_sha,
                "out": str(out),
            },
            indent=2,
        )
    )
    return 0


def _histogram(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def write_run_frames(
    out: Path,
    frames_dir: Path,
    run_index: int,
    stage: int,
    difficulty: int,
    policy: str,
    engine_sha: str,
    raw: list[dict],
) -> int:
    """Label one run's buffered frames and append them to the JSONL."""

    from touhou_rl.v400.labels import (
        LABEL_VERSION,
        encode_occupancy,
        evaluate_future_danger,
        event_labels,
        hazard_occupancy,
        hud_label,
        player_label,
    )
    from touhou_rl.v400.labels.png import write_rgba_png

    if not raw:
        return 0
    horizon = 16
    payloads: list[dict] = []
    for position, item in enumerate(raw):
        successors = [
            frame["hazards"] for frame in raw[position + 1 : position + 1 + horizon]
        ]
        player = item["player"]
        next_state = (
            int(raw[position + 1]["player"].player_state)
            if position + 1 < len(raw)
            else None
        )
        danger = evaluate_future_danger(player, successors, horizon=horizon)
        occupancy = hazard_occupancy(item["hazards"])
        pixels = item.get("pixel_meta")
        if pixels is None:
            name = f"run{run_index}_{item['i']:05d}.png"
            pixels = write_rgba_png(frames_dir / name, item["pixels"])
            pixels["path"] = f"frames/{name}"
            item["pixel_meta"] = pixels
        payloads.append(
            {
                "schema": "v400-p2-label-frame-1",
                "run": run_index,
                "i": item["i"],
                "stage": stage,
                "difficulty": difficulty,
                "policy": policy,
                "mask": item["mask"],
                "frame": item["frame"],
                "shm_frame": item["shm_frame"],
                "game_over": item["game_over"],
                "player": player_label(player),
                "hud": hud_label(item["state"]),
                "events": event_labels(player.player_state, next_state),
                "occupancy": encode_occupancy(occupancy),
                "future_danger": danger.as_dict(),
                "state": item["state"],
                "pixels": pixels,
                "meta": {
                    "label_version": LABEL_VERSION,
                    "engine_sha256": engine_sha,
                    "label_sources": "HAZARDS+PLAYER+STATE only; pixels stored, never read",
                },
            }
        )
    with open(out, "a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
    return len(payloads)


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_overlay_frames(records: list[dict], count: int, seed: int) -> list[dict]:
    """Stratified pick: cover every run, and force the interesting classes in."""

    rng = random.Random(seed)
    by_run: dict[int, list[dict]] = {}
    for record in records:
        by_run.setdefault(int(record["run"]), []).append(record)
    selected: list[dict] = []
    seen: set[int] = set()

    def add(record: dict) -> None:
        key = id(record)
        if key in seen or len(selected) >= count:
            return
        seen.add(key)
        selected.append(record)

    # 1) every run gets a stride sample first, so no run can be squeezed out by
    #    the class-coverage pass.
    per_run = max(1, count // max(1, len(by_run)) // 2)
    for run, rows in sorted(by_run.items()):
        ordered = sorted(rows, key=lambda record: int(record["i"]))
        take = min(per_run, len(ordered))
        if take:
            stride = max(1, len(ordered) // take)
            for index in range(0, len(ordered), stride):
                if len(selected) >= count:
                    break
                add(ordered[index])

    # 2) class coverage: death-proxy edges, future-danger positives, empty grids.
    for predicate, quota in (
        (lambda r: (r.get("events", {}).get("death") or {}).get("value") is True, count // 4),
        (lambda r: (r.get("future_danger") or {}).get("would_die") is True, count // 6),
        (lambda r: int((r.get("occupancy") or {}).get("occupied_cells", 0)) == 0, count // 10),
    ):
        taken = 0
        for record in records:
            if taken >= quota or len(selected) >= count:
                break
            if id(record) in seen or not predicate(record):
                continue
            add(record)
            taken += 1

    # 3) fill remainder round-robin.
    while len(selected) < count:
        before = len(selected)
        for run, rows in sorted(by_run.items()):
            if len(selected) >= count:
                break
            candidates = [r for r in rows if id(r) not in seen]
            if candidates:
                add(rng.choice(candidates))
        if len(selected) == before:
            break
    return selected[:count]


def _gray(pixels):
    import numpy as np

    return pixels[:, :, :3].astype(np.float32).mean(axis=2)


def measure_playfield_origin(frames: list) -> dict:
    """Measure the rendered playfield edge against the (32, 16) claim."""

    import numpy as np

    col_edges = np.zeros(48, dtype=np.float64)
    row_edges = np.zeros(32, dtype=np.float64)
    for pixels in frames:
        gray = _gray(pixels)
        # column edge between x and x+1, over the playfield's vertical span
        for x in range(24, 44):
            col_edges[x] += float(np.abs(gray[16:464, x] - gray[16:464, x + 1]).mean())
        for y in range(8, 28):
            row_edges[y] += float(np.abs(gray[y, 32:416] - gray[y + 1, 32:416]).mean())
    n = max(1, len(frames))
    col_edges /= n
    row_edges /= n
    col_best_index = int(np.argmax(col_edges[24:44])) + 24
    row_best_index = int(np.argmax(row_edges[8:28])) + 8
    # A column edge stored at index x is the boundary *between* x and x+1, so the
    # measured origin is the boundary index plus one.
    measured = [col_best_index + 1, row_best_index + 1]
    return {
        "frames_measured": len(frames),
        "expected_origin": [32, 16],
        "measured_origin": measured,
        "boundary_column_index": col_best_index,
        "boundary_row_index": row_best_index,
        "column_edge_at_32": float(col_edges[32]),
        "column_edge_profile_24_43": [float(col_edges[x]) for x in range(24, 44)],
        "row_edge_at_16": float(row_edges[16]),
        "row_edge_profile_8_27": [float(row_edges[y]) for y in range(8, 28)],
        "column_matches": measured[0] == 32,
        "row_matches": measured[1] == 16,
        "method": (
            "mean |gray(x)-gray(x+1)| over rows 16..463 (columns, x=24..43) and "
            "|gray(y)-gray(y+1)| over cols 32..415 (rows, y=8..27), averaged over "
            "the 50 overlay frames; the playfield gutter/stage boundary is the "
            "strongest edge, and a boundary at index i means origin i+1"
        ),
    }


def measure_bullet_alignment(frames: list[dict], pixels: list) -> dict:
    """Measure the state->screen translation by aligning bullets to bright cores.

    For every lethal-bullet grid cell the label says is *inside* a bullet AABB
    (distance <= 2 px), sample the frame's brightness at ``cell + (32,16) +
    (dx,dy)`` over a +/-8 px search and average over all cells of the busiest
    frames.  The offset that maximises mean brightness is the translation that
    puts state coordinates onto rendered bullets.  This is the empirical half of
    the "screen = playfield + (32,16)" assertion; the grid cell itself is only a
    4 px quantisation, which is why the peak is reported with its profile.
    """

    import numpy as np

    from touhou_rl.v400.labels.occupancy import decode_occupancy

    order = sorted(
        range(len(frames)),
        key=lambda index: -int((frames[index].get("occupancy") or {}).get("bullet_boxes", 0)),
    )[:12]
    accum: dict[tuple[int, int], float] = {}
    used = 0
    cells_total = 0
    for index in order:
        occupancy = frames[index].get("occupancy") or {}
        if int(occupancy.get("bullet_boxes", 0)) < 20:
            continue
        grid = decode_occupancy(occupancy)
        rows, cols = np.nonzero(grid.bullet_occupied)
        if rows.size == 0:
            continue
        keep = grid.distance[rows, cols] <= 2.0
        rows, cols = rows[keep], cols[keep]
        if rows.size == 0:
            continue
        xs = (cols + 0.5) * grid.cell + 32.0
        ys = (rows + 0.5) * grid.cell + 16.0
        image = pixels[index]
        value = image[:, :, :3].astype(np.float32).max(axis=2)
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                px = np.clip((xs + dx).astype(int), 0, 639)
                py = np.clip((ys + dy).astype(int), 0, 479)
                accum[(dx, dy)] = accum.get((dx, dy), 0.0) + float(value[py, px].mean())
        used += 1
        cells_total += int(rows.size)
    if not accum:
        return {"frames": 0}
    best = max(accum.items(), key=lambda item: item[1])
    baseline = float(
        np.mean([value for key, value in accum.items() if abs(key[0]) == 8 or abs(key[1]) == 8])
    )
    return {
        "frames": used,
        "bullet_cells": cells_total,
        "best_offset": [int(best[0][0]), int(best[0][1])],
        "best_mean_brightness": best[1] / max(1, used),
        "edge_baseline_brightness": baseline / max(1, used),
        "peak_over_baseline": (best[1] / max(1, used)) - (baseline / max(1, used)),
        "search_radius_px": 8,
        "method": (
            "mean max(R,G,B) at bullet-cell centres + (32,16) + (dx,dy) over a "
            "+/-8 px grid; the peak offset is the measured state->screen "
            "translation for bullets, averaged over the busiest overlay frames"
        ),
    }


def measure_focus_dot(frames: list[dict], pixels: list) -> dict:
    """Diagnostic: the focused hitbox dot should sit on player+(32,16)."""

    import numpy as np

    offsets: list[tuple[float, float]] = []
    detected = 0
    considered = 0
    for record, image in zip(frames, pixels):
        player = record.get("player") or {}
        if not player.get("focus") or player.get("x") is None:
            continue
        considered += 1
        sx = player["x"] + 32.0
        sy = player["y"] + 16.0
        x0, y0 = int(round(sx)) - 6, int(round(sy)) - 6
        window = image[max(0, y0) : y0 + 13, max(0, x0) : x0 + 13]
        red = (
            (window[:, :, 0] > 170)
            & (window[:, :, 1] < 110)
            & (window[:, :, 2] < 110)
        )
        if red.sum() < 3:
            continue
        ys, xs = np.nonzero(red)
        cx = x0 + xs.mean()
        cy = y0 + ys.mean()
        offsets.append((cx - sx, cy - sy))
        detected += 1
    if offsets:
        arr = np.asarray(offsets, dtype=np.float64)
        summary = {
            "nx": float(arr[:, 0].mean()),
            "ny": float(arr[:, 1].mean()),
            "max_abs": float(np.abs(arr).max()),
        }
    else:
        summary = {"nx": None, "ny": None, "max_abs": None}
    return {
        "considered_focused_frames": considered,
        "detected": detected,
        "mean_offset_px": [summary["nx"], summary["ny"]],
        "max_abs_offset_px": summary["max_abs"],
        "method": (
            "within +/-6 px of player+(32,16), centroid of red-dominant pixels "
            "(r>170, g<110, b<110).  This centroid is the *sprite-space* focus "
            "indicator, not the state point: the authoritative conversion check "
            "is bullet_alignment, which peaks on rendered bullet cores.  A "
            "non-zero offset here records how far the drawn indicator sits from "
            "the state point, not an error in screen = playfield + (32,16)"
        ),
    }


def measure_occupancy_alignment(frames: list[dict], pixels: list) -> dict:
    """Do the state-derived bullet cells land on bright rendered bullets?"""

    import numpy as np

    ratios = []
    for record, image in zip(frames, pixels):
        occupancy = record.get("occupancy") or {}
        if not occupancy:
            continue
        from touhou_rl.v400.labels.occupancy import decode_occupancy

        grid = decode_occupancy(occupancy)
        bright = (
            (image[:, :, 0] > 200)
            & (image[:, :, 1] > 200)
            & (image[:, :, 2] > 200)
        )
        play = bright[16:464, 32:416]
        mask = np.repeat(
            np.repeat(grid.bullet_occupied, grid.cell, axis=0), grid.cell, axis=1
        )
        inside = play[mask]
        if inside.size == 0:
            continue
        outside = play[~mask]
        if outside.size == 0:
            continue
        ratios.append(float(inside.mean() - outside.mean()))
    if not ratios:
        return {"frames": 0}
    return {
        "frames": len(ratios),
        "mean_density_gain_inside_bullet_cells": float(np.mean(ratios)),
        "min": float(np.min(ratios)),
        "max": float(np.max(ratios)),
        "note": (
            "white-pixel density inside state-derived bullet cells minus outside; "
            "positive means the cells sit on rendered bright bullet cores. "
            "Background art adds noise, so this is a diagnostic, not a gate"
        ),
    }


def overlay(args) -> int:
    import numpy as np
    from PIL import Image

    from touhou_rl.v400.labels.overlay import (
        load_png,
        render_montage,
        render_overlay,
    )
    from touhou_rl.v400.labels.spec import SCREEN_ORIGIN

    labels_path = Path(args.labels)
    records = _load_jsonl(labels_path)
    if not records:
        raise SystemExit(f"no frames in {labels_path}")
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = select_overlay_frames(records, args.count, args.seed)
    if len(picked) < args.count:
        print(
            f"warning: only {len(picked)} frames available, asked for {args.count}",
            file=sys.stderr,
        )

    overlays = []
    index_rows = []
    raw_pixels = []
    for position, record in enumerate(picked):
        png_path = frames_dir / record["pixels"]["file"]
        pixels = load_png(png_path)
        raw_pixels.append(pixels)
        image = render_overlay(pixels, record, origin=SCREEN_ORIGIN)
        name = f"overlay-{position:02d}-run{record['run']}-f{record['frame']}.png"
        image.save(out_dir / name)
        overlays.append(image)
        index_rows.append(
            {
                "overlay": name,
                "run": record["run"],
                "frame": record["frame"],
                "policy": record["policy"],
                "stage": record["stage"],
                "death_proxy": (record.get("events", {}).get("death") or {}).get("value"),
                "future_danger": (record.get("future_danger") or {}).get("would_die"),
                "first_hit_offset": (record.get("future_danger") or {}).get("first_hit_offset"),
                "min_gap": (record.get("future_danger") or {}).get("min_gap"),
                "occupied_cells": (record.get("occupancy") or {}).get("occupied_cells"),
                "bullet_cells": (record.get("occupancy") or {}).get("bullet_cells"),
                "player": [record.get("player", {}).get("x"), record.get("player", {}).get("y")],
                "hud_text": (record.get("hud") or {}).get("text"),
            }
        )
    montage = render_montage(overlays, columns=args.columns, scale=args.scale)
    montage_path = out_dir / "montage.png"
    montage.save(montage_path)

    coordinates = {
        "screen_origin": list(SCREEN_ORIGIN),
        "spec_claim": "screen = playfield + (32, 16)",
        "source_evidence": (
            "third_party/th08-rl/src/Background.cpp:740-743 sets "
            "viewport.X=32, viewport.Y=16, Width=384, Height=448"
        ),
        "measured_playfield_origin": measure_playfield_origin(raw_pixels),
        "bullet_alignment": measure_bullet_alignment(picked, raw_pixels),
        "focus_dot": measure_focus_dot(picked, raw_pixels),
        "occupancy_vs_pixels": measure_occupancy_alignment(picked, raw_pixels),
        "overlay_count": len(overlays),
        "montage": str(montage_path),
    }
    (out_dir / "coords.json").write_text(
        json.dumps(coordinates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "labels": str(labels_path),
                "labels_sha256": sha256_file(labels_path),
                "count": len(overlays),
                "montage": str(montage_path),
                "frames": index_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "overlays": len(overlays),
                "montage": str(montage_path),
                "coords": coordinates["measured_playfield_origin"],
                "focus_dot": coordinates["focus_dot"],
                "occupancy_alignment": coordinates["occupancy_vs_pixels"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def audit_command(args) -> int:
    from touhou_rl.v400.labels.audit import audit, hud_lag, to_markdown

    labels_path = Path(args.labels)
    records = _load_jsonl(labels_path)
    if not records:
        raise SystemExit(f"no frames in {labels_path}")
    report = audit(records, frames_dir=args.frames_dir)
    report["dataset"]["path"] = str(labels_path)
    report["dataset"]["sha256"] = sha256_file(labels_path)
    report["dataset"]["meta"] = str(labels_path.with_suffix(".meta.json"))
    if args.lag and args.frames_dir:
        report["hud_lag"] = hud_lag(records, args.frames_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = to_markdown(report)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(json.dumps({"passes": report["passes"], "out": str(out)}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    collect_parser = sub.add_parser("collect", help="engine side: frames + labels")
    collect_parser.add_argument("--engine", required=True)
    collect_parser.add_argument("--data-dir", required=True)
    collect_parser.add_argument("--out", required=True, help="labels-*.jsonl")
    collect_parser.add_argument("--frames-dir", required=True)
    collect_parser.add_argument("--role", default="env")
    collect_parser.add_argument("--seed", type=int, default=1)
    collect_parser.add_argument("--hold", type=int, default=6)
    collect_parser.add_argument("--warmup", type=int, default=60)
    collect_parser.add_argument("--flush-every", type=int, default=0)
    collect_parser.add_argument("--timeout", type=float, default=300.0)
    collect_parser.add_argument("--runs", nargs="+", required=True)
    collect_parser.set_defaults(func=collect)

    overlay_parser = sub.add_parser("overlay", help="draw 50 overlays + montage")
    overlay_parser.add_argument("--labels", required=True)
    overlay_parser.add_argument("--frames-dir", required=True)
    overlay_parser.add_argument("--out-dir", required=True)
    overlay_parser.add_argument("--count", type=int, default=50)
    overlay_parser.add_argument("--seed", type=int, default=20261109)
    overlay_parser.add_argument("--columns", type=int, default=5)
    overlay_parser.add_argument("--scale", type=float, default=0.5)
    overlay_parser.set_defaults(func=overlay)

    audit_parser = sub.add_parser("audit", help="audit table + optional HUD lag")
    audit_parser.add_argument("--labels", required=True)
    audit_parser.add_argument("--frames-dir")
    audit_parser.add_argument("--out", required=True)
    audit_parser.add_argument("--lag", action="store_true")
    audit_parser.set_defaults(func=audit_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
