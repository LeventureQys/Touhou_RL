#!/usr/bin/env python3
"""Stage-clear mass production from the 108 official replays (EXP-P2-008).

The Linux harness build ``rl_events`` replays a real ``.rpy`` through the
developer-autoplay path (``TH08_AUTOPLAY_REPLAY`` / ``TH08_AUTOPLAY_STAGE``) and
reports the harness-derived ``STAGE_CLEAR`` event bit (EVENTS bit ``0x0080``)
plus the exact ``stageClearEdgeFrame``.  This tool turns that into a labelled
pixel dataset around every real stage-clear edge:

* ``plan``   reads the per-stage frame counts already measured by the WP1
             differential retrace run (``.tmp_v400_replay_retrace_108``) and
             picks the cheapest stages first, so a bounded wall-clock budget
             buys the largest number of real events.
* ``worker`` drives one harness process per event on an isolated data dir,
             replays only that stage (``TH08_AUTOPLAY_STAGE=<stage>``), turns
             the per-frame readback on shortly before the predicted clear,
             and keeps a rolling window of ``play64`` / ``hud`` / thumbnail
             planes plus scalar screen statistics.  When the engine's own
             ``stageClearEdgeFrame`` lands inside the window, the
             ``edge-60 .. edge+120`` slice is written out; anything else is a
             fail-closed journal record, never a fabricated event.
* ``report`` folds the worker journals into ``report.json`` / ``summary.json``.

The stage's own frames are the only unavoidable cost: replay playback is
sequential, so a stage clear cannot be reached without simulating the stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import struct
import subprocess
import sys
import time
from pathlib import Path

WIDTH = 640
HEIGHT = 480
PIXELS = WIDTH * HEIGHT * 4
SHM_PIXELS_OFFSET = 192
SHM_MAGIC = 0x30303456
SHM_VERSION = 1
HEADER = struct.Struct("<IIQQIIII")
PLAY_ORIGIN = (32, 16)
PLAY_SHAPE = (384, 448)
PLAY64 = (64, 64)
HUD_BANDS = ((48, 72), (80, 96), (96, 112))
HUD_X0, HUD_X1 = 480, 640
THUMB = (32, 32)
PRE_FRAMES = 60
POST_FRAMES = 120
WINDOW = PRE_FRAMES + POST_FRAMES + 1
RING = 420
GRAY = (299, 587, 114)


class ReplayError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# harness client
# ---------------------------------------------------------------------------
class Harness:
    def __init__(self, argv: list[str], env: dict[str, str]):
        self.argv = argv
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, bufsize=0)
        self.buffer = b""
        self.replies = 0

    def ask(self, command: str, timeout: float = 180.0) -> dict[str, str]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write((command + "\n").encode("ascii"))
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ReplayError(f"timeout on {command!r} exit={self.proc.poll()}")
            ready, _, _ = select.select([self.proc.stdout], [], [], min(remaining, 0.25))
            if ready:
                chunk = os.read(self.proc.stdout.fileno(), 1 << 16)
                if not chunk:
                    raise ReplayError(f"pipe EOF on {command!r} exit={self.proc.poll()}")
                self.buffer += chunk
            elif self.proc.poll() is not None:
                raise ReplayError(f"engine exited on {command!r} exit={self.proc.returncode}")
        line, self.buffer = self.buffer.split(b"\n", 1)
        reply = line.decode("utf-8", "replace").strip()
        if not reply.startswith("OK "):
            raise ReplayError(f"{command!r} rejected: {reply}")
        self.replies += 1
        return dict(part.split("=", 1) for part in reply.split()[1:] if "=" in part)

    def close(self) -> dict[str, object]:
        code = None
        if self.proc.poll() is None:
            try:
                self.ask("QUIT", timeout=15.0)
            except (ReplayError, OSError):
                pass
        try:
            code = self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = self.proc.wait(timeout=5)
        stderr = b""
        if self.proc.stderr is not None:
            try:
                stderr = self.proc.stderr.read() or b""
            except OSError:
                stderr = b""
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        return {"exitCode": code, "stderrTail": stderr.decode("utf-8", "replace")[-1500:]}


# ---------------------------------------------------------------------------
# pixel planes (byte-exact with touhou_rl.v400.student.obs_spec.encode_planes)
# ---------------------------------------------------------------------------
def encode_frame(np, rgba: bytes) -> dict:
    if len(rgba) != PIXELS:
        raise ReplayError(f"readback is {len(rgba)} bytes, expected {PIXELS}")
    frame = np.frombuffer(rgba, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[::-1]
    x0, y0 = PLAY_ORIGIN
    width, height = PLAY_SHAPE
    crop = frame[y0:y0 + height, x0:x0 + width]
    red = crop[:, :, 0].astype(np.uint16)
    green = crop[:, :, 1].astype(np.uint16)
    blue = crop[:, :, 2].astype(np.uint16)
    lum = ((red * GRAY[0] + green * GRAY[1] + blue * GRAY[2]) >> 10).astype(np.uint32)
    oh, ow = PLAY64
    bh, bw = height // oh, width // ow
    play = (lum.reshape(oh, bh, ow, bw).sum(axis=(1, 3)) // (bh * bw)).astype(np.uint8)
    hud = np.concatenate([frame[a:b, HUD_X0:HUD_X1, 1] for (a, b) in HUD_BANDS], axis=0)
    thumb = (frame[:, :, :3].astype(np.uint16)
             .reshape(THUMB[0], HEIGHT // THUMB[0], THUMB[1], WIDTH // THUMB[1], 3)
             .mean(axis=(1, 3))).astype(np.uint8)
    return {"play": play, "hud": hud, "thumb": thumb, "lum": lum, "frame": frame}


def frame_stats(np, planes: dict, previous: np.ndarray | None) -> list[float]:
    lum = planes["lum"].astype(np.float32)
    rgb = planes["frame"][:, :, :3].astype(np.float32)
    play = planes["play"].astype(np.float32)
    gx = np.abs(np.diff(lum[:, ::2], axis=1)).mean()
    gy = np.abs(np.diff(lum[::2, :], axis=0)).mean()
    colour = (np.abs(rgb[:, :, 0] - rgb[:, :, 1]).mean()
              + np.abs(rgb[:, :, 1] - rgb[:, :, 2]).mean())
    motion = 0.0 if previous is None else float(
        np.abs(lum - previous).mean())
    return [
        float(lum.mean()), float(lum.std()), float(play.mean()), float(play.std()),
        float(gx), float(gy), float(colour), float(motion),
        float((lum > 200).mean()), float((lum < 32).mean()),
        float(np.abs(np.diff(lum, axis=1)).mean()), float(np.abs(np.diff(lum, axis=0)).mean()),
        float(rgb[:, :, 0].mean()), float(rgb[:, :, 1].mean()), float(rgb[:, :, 2].mean()),
        float(planes["hud"].astype(np.float32).mean()),
    ]


HUD_KEYS = ("displayScore", "globalsScore", "lives", "bombs", "power", "graze",
            "point", "time", "gameOverFlag", "displayAge")


def parse_hud(reply: dict[str, str]) -> list[float]:
    values = []
    for key in HUD_KEYS:
        try:
            values.append(float(reply.get(key, "nan")))
        except ValueError:
            values.append(float("nan"))
    return values


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
RETRACE_CASE_COMPLETE = "on/attempt-1/complete.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_plan(args) -> int:
    retrace = Path(args.retrace_dir)
    fixtures = Path(args.fixtures_dir)
    records: dict[str, dict] = {}
    for shard in sorted(retrace.glob("shard-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["case"]] = record
    events = []
    for case, record in sorted(records.items()):
        complete_path = retrace / case / RETRACE_CASE_COMPLETE
        if not complete_path.is_file():
            continue
        complete = load_json(complete_path)
        stage_frames = {int(k): int(v) for k, v in (complete.get("stageFrames") or {}).items()}
        # The retrace inventory records whatever separator its host used; the
        # plan has to be replayable from WSL (POSIX) *and* from Windows, so try
        # the literal path, the separator-normalised path, then the fixtures dir.
        recorded = Path(record["replay"]["path"])
        normalised = Path(str(record["replay"]["path"]).replace("\\", "/"))
        replay_path = recorded
        for candidate in (recorded, normalised):
            if candidate.is_file():
                replay_path = candidate
                break
        if not replay_path.is_file():
            alternative = fixtures / Path(normalised.name)
            replay_path = alternative if alternative.is_file() else replay_path
        if not replay_path.is_file():
            continue
        for stage, frames in sorted(stage_frames.items()):
            if frames <= 0:
                continue
            events.append({
                "case": case,
                "fixtureId": record.get("fixtureId"),
                "difficulty": record["replay"]["difficulty"],
                "stage": stage,
                "stageFrames": frames,
                "replay": str(replay_path),
                "replaySha256": record["replay"]["sha256"],
            })
    events.sort(key=lambda item: (item["stageFrames"], item["case"], item["stage"]))
    selected = events[:args.target_events]
    cumulative = 0
    for index, event in enumerate(selected):
        cumulative += event["stageFrames"]
        event["order"] = index
        event["cumulativeFrames"] = cumulative
    plan = {
        "schema": "v400-stage-clear-plan/1",
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "retraceDir": str(retrace),
        "fixturesDir": str(fixtures),
        "targetEvents": args.target_events,
        "availableEvents": len(events),
        "availableFrames": sum(item["stageFrames"] for item in events),
        "selectedEvents": len(selected),
        "selectedFrames": cumulative,
        "estimatedSecondsAt140fps": round(cumulative / 140.0, 1),
        "events": selected,
        # The full parseable universe (all stage entries of the 108 replays),
        # carried so the coverage report can zero-fill a difficulty/stage the
        # cheapest-first selection never reached (EXP-P2-009 §coverage).
        "available": [{"case": item["case"], "difficulty": item["difficulty"],
                       "stage": item["stage"], "stageFrames": item["stageFrames"]}
                      for item in events],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: plan[k] for k in ("availableEvents", "availableFrames",
                                           "selectedEvents", "selectedFrames",
                                           "estimatedSecondsAt140fps")}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------
def setup_data_dir(path: Path, assets: Path, template: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "backup").mkdir(exist_ok=True)
    for asset in ("th08.dat", "thbgm.dat"):
        link = path / asset
        if not link.exists():
            link.symlink_to(assets / asset)
    for writable in ("th08.cfg", "score.dat"):
        copy = path / writable
        if not copy.exists() or copy.stat().st_size == 0:
            source = template / writable
            copy.write_bytes(source.read_bytes())


def event_data_dir(data_root: Path, event: dict) -> Path:
    """One data dir per **event**, never per worker.

    EXP-P2-009 tightened the isolation rule: a worker reuses its process slot
    but every event gets its own ``--data-dir`` so no ``score.dat`` / config
    write from a previous event can leak into the next one.
    """

    return data_root / f"{event['case']}-s{int(event['stage'])}"


def captured_keys(out_dirs: list[str]) -> set[tuple[str, int]]:
    """Resume: every (case, stage) already captured *and committed to disk*.

    A journal line alone is not enough — the ``frames.npz`` must still exist,
    otherwise the event is re-attempted (fail-closed both ways: no phantom
    skip, no phantom event).
    """

    seen: set[tuple[str, int]] = set()
    for out in out_dirs:
        root = Path(out)
        for journal in sorted(root.glob("journal-*.jsonl")):
            for line in journal.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("status") != "captured":
                    continue
                key = (record["case"], int(record["stage"]))
                if (root / "events" / f"{key[0]}-s{key[1]}" / "frames.npz").is_file():
                    seen.add(key)
    return seen


def append_journal(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def predicted_capture_start(startup: int, stage_frames: int, pre_roll: int = 600) -> int:
    """Harness frame at which per-frame readback turns on.

    The prediction only has to be *early*: the ring buffer keeps ``RING``
    frames, and the event is accepted only when the engine's own edge frame
    lands inside it.  ``stage_frames`` is the exact gameplay-frame count
    measured by the retrace run, and the harness adds stage setup frames, so
    a ``pre_roll``-frame pre-roll plus the ring is generous.

    EXP-P2-008 lost two events because the real edge landed 11 and 35 frames
    *before* a 600-frame pre-roll; EXP-P2-009 defaults to 900 so those land
    inside the ring.  The pre-roll only widens *readback*, never the labelled
    window (``[edge-60, edge+120]`` is unchanged).
    """

    return max(startup + 4, startup + stage_frames - int(pre_roll))


def capture_event(np, args, event: dict, out_root: Path, deadline: float) -> dict:
    replay = Path(event["replay"])
    stage = int(event["stage"])
    if not replay.is_file():
        raise ReplayError(f"replay missing: {replay}")
    data_dir = event_data_dir(Path(args.data_root), event)
    setup_data_dir(data_dir, Path(args.assets), Path(args.template))
    record_data_dir = str(data_dir)
    shm_name = f"v400sc{os.getpid()}"
    shm_path = Path("/dev/shm") / shm_name
    if shm_path.exists():
        shm_path.unlink()
    env = dict(os.environ)
    env.update({
        "TH08_AUTOPLAY_REPLAY": str(replay),
        "TH08_AUTOPLAY_STAGE": str(stage),
        "SDL_AUDIODRIVER": "dummy",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "LP_NUM_THREADS": "1",
        "TH08_DISABLE_CRASH_REPORTER": "1",
    })
    argv = [args.binary, "--data-dir", str(data_dir), "--rl-control", "0,1",
            "--rl-role", "env"]
    started = time.time()
    client = Harness(argv, env)
    record: dict = {
        "status": "failed",
        "case": event["case"],
        "fixtureId": event.get("fixtureId"),
        "difficulty": event.get("difficulty"),
        "stage": stage,
        "stageFrames": event["stageFrames"],
        "replay": str(replay),
        "replaySha256": event["replaySha256"],
        "binary": args.binary,
        "binarySha256": args.binary_sha256,
        "dataDir": record_data_dir,
        "startedUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    }
    frames: list[dict] = []
    try:
        hello = client.ask("HELLO", timeout=180)
        if hello.get("role") != "env":
            raise ReplayError(f"HELLO role {hello.get('role')!r} != 'env'")
        client.ask("READBACK 0")
        shm = client.ask("SHM /" + shm_name)
        if shm.get("active") != "1" or int(shm.get("bytes", -1)) != SHM_PIXELS_OFFSET + PIXELS:
            raise ReplayError(f"shared block unavailable: {shm}")
        startup = 0
        debug: dict[str, str] = client.ask("DEBUG")
        while startup < args.startup_guard:
            debug = client.ask("DEBUG")
            if debug.get("isReplay") == "1" and debug.get("globals") == "1":
                break
            client.ask("STEP 0 1")
            startup += 1
        if debug.get("isReplay") != "1":
            raise ReplayError(
                f"TH08_AUTOPLAY_REPLAY never armed within {args.startup_guard} frames: {debug}")
        record["startupFrames"] = startup
        record["debugStart"] = debug

        capture_start = predicted_capture_start(startup, event["stageFrames"], args.pre_roll)
        record["captureStart"] = capture_start
        frame = startup
        while frame < capture_start:
            step = min(args.scan_chunk, capture_start - frame)
            reply = client.ask(f"STEP 0 {step}")
            advanced = int(reply.get("advanced", "0"))
            if advanced < step:
                raise ReplayError(f"playback exited early before capture window: {reply}")
            frame += advanced
            events = client.ask("EVENTS")
            if int(events.get("stageClear", "0")) == 1:
                raise ReplayError(
                    "stage clear already up before the readback window "
                    f"(frame={events.get('frames')} edge={events.get('stageClearEdgeFrame')})")
        record["frameAtCaptureStart"] = frame
        client.ask("READBACK 1")
        ring: list[dict] = []
        previous = None
        edge = None
        edge_frame = None
        while True:
            if time.time() > deadline + 90:
                raise ReplayError("hard deadline reached inside capture window")
            reply = client.ask("STEP 0 1")
            if int(reply.get("advanced", "0")) < 1:
                raise ReplayError(f"playback exited inside capture window: {reply}")
            frame += 1
            events = client.ask("EVENTS")
            hud = parse_hud(client.ask("HUD"))
            header, rgba = read_shm(shm_path)
            # PumpOneFrame() publishes the readback inside the frame, before
            # ++g_frameCount, so the shared header always labels the pixel
            # buffer one frame below the EVENTS reply.  Accept only that exact
            # relation; anything else means the two reads straddled a frame.
            if int(header["frame"]) != int(events.get("frames", "-1")) - 1:
                header, rgba = read_shm(shm_path)
                if int(header["frame"]) != int(events.get("frames", "-1")) - 1:
                    raise ReplayError(
                        f"pixel frame {header['frame']} != EVENTS frame {events.get('frames')} - 1")
            planes = encode_frame(np, rgba)
            stats = frame_stats(np, planes, previous)
            previous = planes["lum"].astype(np.float32)
            level = int(events.get("stageClear", "0"))
            entry = {
                "frame": frame,
                "level": level,
                "screenState": int(events.get("stageClearScreenState", "-1")),
                "seq": int(events.get("stageClearSeq", "0")),
                "flags": int(events.get("frameEventFlags", "0"), 16),
                "play": planes["play"],
                "hud": planes["hud"],
                "thumb": planes["thumb"],
                "stats": stats,
                "hudFields": hud,
            }
            ring.append(entry)
            if len(ring) > RING:
                ring.pop(0)
            if edge is None and level == 1:
                edge = frame
                edge_frame = int(events.get("stageClearEdgeFrame", "0"))
            if edge is not None and frame - edge >= POST_FRAMES:
                break
            if edge is None and frame - capture_start > args.max_window:
                raise ReplayError(
                    f"no stage clear within {frame - capture_start} frames after window start")
        record["edgeFrame"] = edge
        record["stageClearEdgeFrame"] = edge_frame
        if edge_frame != edge:
            raise ReplayError(
                f"latched edge {edge_frame} != observed rising edge {edge}")
        window = [entry for entry in ring if edge - PRE_FRAMES <= entry["frame"] <= edge + POST_FRAMES]
        have = {entry["frame"] for entry in window}
        missing = [index for index in range(edge - PRE_FRAMES, edge + POST_FRAMES + 1)
                   if index not in have]
        if missing:
            raise ReplayError(f"window incomplete, missing {len(missing)} frames incl {missing[:5]}")
        window.sort(key=lambda entry: entry["frame"])
        event_dir = out_root / "events" / f"{event['case']}-s{stage}"
        event_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "frame": np.asarray([entry["frame"] for entry in window], dtype=np.int64),
            "level": np.asarray([entry["level"] for entry in window], dtype=np.uint8),
            "screenState": np.asarray([entry["screenState"] for entry in window], dtype=np.int16),
            "seq": np.asarray([entry["seq"] for entry in window], dtype=np.int32),
            "flags": np.asarray([entry["flags"] for entry in window], dtype=np.uint32),
            "play": np.stack([entry["play"] for entry in window]),
            "hud": np.stack([entry["hud"] for entry in window]),
            "thumb": np.stack([entry["thumb"] for entry in window]),
            "stats": np.asarray([entry["stats"] for entry in window], dtype=np.float32),
            "hudFields": np.asarray([entry["hudFields"] for entry in window], dtype=np.float32),
        }
        out_path = event_dir / "frames.npz"
        np.savez_compressed(out_path, **arrays)
        meta = {
            "schema": "v400-stage-clear-event/1",
            "status": "captured",
            "case": event["case"],
            "fixtureId": event.get("fixtureId"),
            "difficulty": event.get("difficulty"),
            "stage": stage,
            "stageFrames": event["stageFrames"],
            "replay": str(replay),
            "replaySha256": event["replaySha256"],
            "binary": args.binary,
            "binarySha256": args.binary_sha256,
            "startupFrames": startup,
            "captureStart": capture_start,
            "edgeFrame": edge,
            "stageClearEdgeFrame": edge_frame,
            "window": [edge - PRE_FRAMES, edge + POST_FRAMES],
            "frames": int(arrays["frame"].size),
            "preFrames": PRE_FRAMES,
            "postFrames": POST_FRAMES,
            "statsKeys": ["lumMean", "lumStd", "playMean", "playStd", "gradX", "gradY",
                          "colour", "motion", "brightFrac", "darkFrac",
                          "absDx", "absDy", "rMean", "gMean", "bMean", "hudMean"],
            "hudKeys": list(HUD_KEYS),
            "planes": {"play": [int(v) for v in arrays["play"].shape],
                       "hud": [int(v) for v in arrays["hud"].shape],
                       "thumb": [int(v) for v in arrays["thumb"].shape]},
            "pixelEncoding": "obs_spec.encode_planes (play64 luminance block means, hud green bands)",
            "npzSha256": sha256_file(out_path),
            "seconds": round(time.time() - started, 3),
        }
        (event_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        record.update({"status": "captured", "eventDir": str(event_dir),
                       "npzSha256": meta["npzSha256"], "frames": meta["frames"]})
    except ReplayError as exc:
        record["error"] = str(exc)
    finally:
        close = client.close()
        record["process"] = close
        record["seconds"] = round(time.time() - started, 3)
        record["endedUtc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if shm_path.exists():
            try:
                shm_path.unlink()
            except OSError:
                pass
    return record


def read_shm(path: Path) -> tuple[dict[str, int], bytes]:
    raw = path.read_bytes()
    if len(raw) != SHM_PIXELS_OFFSET + PIXELS:
        raise ReplayError(f"shared block size {len(raw)} != {SHM_PIXELS_OFFSET + PIXELS}")
    magic, version, frame, state_hash, width, height, state_bytes, flags = HEADER.unpack_from(raw)
    if (magic, version, width, height) != (SHM_MAGIC, SHM_VERSION, WIDTH, HEIGHT):
        raise ReplayError(f"shared header invalid: {hex(magic)} v{version} {width}x{height}")
    if not flags & (1 << 1):
        raise ReplayError("shared block carries no pixel publication flag")
    return {"frame": frame, "stateHash": state_hash, "flags": flags}, raw[SHM_PIXELS_OFFSET:]


def run_worker(args) -> int:
    import numpy as np

    plan = load_json(Path(args.plan))
    events = plan["events"]
    out_root = Path(args.out)
    (out_root / "events").mkdir(parents=True, exist_ok=True)
    journal = out_root / f"journal-{int(args.index):02d}.jsonl"
    deadline = float(args.deadline_epoch)
    skip = captured_keys(list(args.resume_dir or []))
    mine = [event for event in events
            if event["order"] % int(args.instances) == int(args.index)
            and (event["case"], int(event["stage"])) not in skip]
    print(json.dumps({"worker": int(args.index), "resumeSkip": len(skip),
                      "assigned": len(mine)}), flush=True)
    done = int(args.max_events) or 0
    for event in mine:
        if time.time() > deadline:
            break
        estimate = float(event["stageFrames"]) / 110.0 + 12.0
        if time.time() + estimate > deadline + 120:
            continue
        record = capture_event(np, args, event, out_root, deadline)
        record["worker"] = int(args.index)
        record["replies"] = 0
        append_journal(journal, record)
        done += 1 if record["status"] == "captured" else 0
        print(json.dumps({"worker": int(args.index), "case": event["case"],
                          "stage": event["stage"], "status": record["status"],
                          "seconds": record["seconds"], "error": record.get("error")}),
              flush=True)
        if int(args.max_events) and done >= int(args.max_events):
            break
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
DIFFICULTIES = ("Easy", "Normal", "Hard", "Lunatic", "Extra")
STAGES = (0, 1, 2, 3, 4, 5, 6, 7, 8)


def coverage_block(captured: list[dict], plan: dict | None) -> dict:
    """Difficulty / stage coverage, zero-filled against the full corpus.

    ``available`` counts come from the plan file's ``available`` universe (all
    parseable stage entries of the 108 official replays, not just the ones the
    cheapest-first selection reached), so a difficulty or stage that never got
    sampled shows up as an explicit ``0`` instead of being silently absent.
    """

    available: dict[str, int] = {}
    available_frames: dict[str, int] = {}
    if plan is not None:
        for item in plan.get("available") or []:
            key = f"{item['difficulty']}/s{int(item['stage'])}"
            available[key] = available.get(key, 0) + 1
            available_frames[key] = available_frames.get(key, 0) + int(item["stageFrames"])
    difficulties: dict[str, dict] = {}
    for name in DIFFICULTIES:
        raised = [r for r in captured if r.get("difficulty") == name]
        avail = sum(v for k, v in available.items() if k.startswith(name + "/"))
        difficulties[name] = {
            "captured": len(raised),
            "available": avail if plan is not None else None,
            "frames": sum(int(r.get("stageFrames") or 0) for r in raised),
        }
    stages: dict[str, dict] = {}
    for stage in STAGES:
        raised = [r for r in captured if int(r["stage"]) == stage]
        avail = sum(v for k, v in available.items() if k.endswith(f"/s{stage}"))
        stages[str(stage)] = {
            "captured": len(raised),
            "available": avail if plan is not None else None,
            "frames": sum(int(r.get("stageFrames") or 0) for r in raised),
        }
    cross: dict[str, int] = {}
    for record in captured:
        key = f"{record.get('difficulty')}/s{int(record['stage'])}"
        cross[key] = cross.get(key, 0) + 1
    return {
        "schema": "v400-stage-clear-coverage/1",
        "capturedEvents": len(captured),
        "difficulties": difficulties,
        "stages": stages,
        "difficultyStage": dict(sorted(cross.items())),
        "zeroDifficulties": [n for n in DIFFICULTIES if difficulties[n]["captured"] == 0],
        "zeroStages": [str(s) for s in STAGES if stages[str(s)]["captured"] == 0],
        "availableUniverse": len(available) if plan is not None else None,
        "availableFramesByKey": dict(sorted(available_frames.items())) if plan else {},
    }


def run_report(args) -> int:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    for directory in [out_root] + [Path(p) for p in (args.extra_out or [])]:
        for journal in sorted(directory.glob("journal-*.jsonl")):
            for line in journal.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    captured = [r for r in records if r["status"] == "captured"]
    failures: dict[str, int] = {}
    for record in records:
        if record["status"] != "captured":
            reason = record.get("error", "unknown").split(":")[0][:80]
            failures[reason] = failures.get(reason, 0) + 1
    by_difficulty: dict[str, int] = {}
    for record in captured:
        by_difficulty[record["difficulty"]] = by_difficulty.get(record["difficulty"], 0) + 1
    total_frames = sum(int(r.get("stageFrames") or 0) for r in records)
    plan = load_json(Path(args.plan)) if args.plan else None
    coverage = coverage_block(captured, plan)
    report = {
        "schema": "v400-stage-clear-production/1",
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attempts": len(records),
        "captured": len(captured),
        "failed": len(records) - len(captured),
        "failureReasons": failures,
        "byDifficulty": by_difficulty,
        "distinctStages": len({(r["case"], r["stage"]) for r in captured}),
        "engineSecondsSum": round(sum(float(r.get("seconds") or 0) for r in records), 1),
        "stageFramesSum": total_frames,
        "effectiveFps": round(total_frames / max(1e-9, sum(float(r.get("seconds") or 0)
                                                           for r in records)), 1),
        "coverage": coverage,
        "records": records,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    summary = {k: report[k] for k in ("attempts", "captured", "failed", "failureReasons",
                                      "byDifficulty", "distinctStages", "stageFramesSum",
                                      "engineSecondsSum", "effectiveFps")}
    summary["coverage"] = coverage
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_root / "coverage.json").write_text(
        json.dumps(coverage, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "coverage"}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="select the cheapest stages from the retrace inventory")
    plan.add_argument("--retrace-dir", default=".tmp_v400_replay_retrace_108")
    plan.add_argument("--fixtures-dir",
                      default="third_party/th08/tests/fixtures/replays/th08-1.00d/accepted")
    plan.add_argument("--target-events", type=int, default=386)
    plan.add_argument("--output", default=".tmp_v400_sclear/plan.json")
    plan.set_defaults(func=build_plan)

    worker = sub.add_parser("worker", help="capture one shard of the plan")
    worker.add_argument("--plan", required=True)
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--instances", type=int, default=6)
    worker.add_argument("--binary", required=True)
    worker.add_argument("--assets", default="/root/v400-data/game")
    worker.add_argument("--template", default="/root/v400-data/game")
    worker.add_argument("--data-root", default="/root/v400-data/sclear")
    worker.add_argument("--out", required=True)
    worker.add_argument("--deadline-epoch", type=float, required=True)
    worker.add_argument("--scan-chunk", type=int, default=120)
    worker.add_argument("--startup-guard", type=int, default=600)
    worker.add_argument("--max-window", type=int, default=1500)
    worker.add_argument("--max-events", type=int, default=0)
    worker.add_argument("--pre-roll", type=int, default=900,
                        help="frames of per-frame readback before the predicted edge")
    worker.add_argument("--resume-dir", action="append", default=[],
                        help="prior output dir; its captured (case, stage) pairs are skipped")
    worker.add_argument("--binary-sha256", default=None)
    worker.set_defaults(func=run_worker)

    report = sub.add_parser("report", help="fold worker journals")
    report.add_argument("--out", required=True)
    report.add_argument("--extra-out", action="append", default=[],
                        help="additional output dir whose journals join the fold")
    report.add_argument("--plan", default=None,
                        help="plan file supplying the available difficulty/stage universe")
    report.set_defaults(func=run_report)

    args = parser.parse_args(argv)
    if args.command == "worker" and not args.binary_sha256:
        args.binary_sha256 = sha256_file(Path(args.binary))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
