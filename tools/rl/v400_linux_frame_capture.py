"""Real Linux-harness frame capture for the v4.0.0 WP8 paired-pixel audit.

Two stages, because the two hosts cannot share a Python environment:

``capture``  runs inside WSL/Linux next to the real ``th08-modern`` ELF.  It
             drives the ``--rl-role env`` harness over the line protocol, plays
             a real ``.rpy`` through ``TH08_AUTOPLAY_REPLAY``, seeks to each
             requested ``logicFrame`` (the engine's own
             ``g_GameManager.gameplayFrameCounter``, the same integer the
             Windows fence publishes as ``logicFrame``), and stores the *exact*
             shared-memory RGBA bytes plus the 25 tracked state fields.
``convert``  runs on Windows: it turns the raw RGBA into lossless RGBA PNGs and
             writes ``linux-manifest.json`` for ``v400_pixel_pair.py``.

Orientation: OpenGL returns rows bottom-up, so the PNG writer flips rows.  That
is the same normalization ``HandlePixels`` applies when it writes a PPM; the
raw bytes are kept on disk and their SHA-256 is recorded, so the flip is
auditable instead of hidden.

Nothing here invents a frame: a target that the engine never reaches is an
error, and a shared-memory header whose frame counter does not match the
``STATE`` reply is an error.
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
PIXEL_BYTES = WIDTH * HEIGHT * 4
STATE_BYTES = 25 * 4
SHM_STATE_OFFSET = 64
SHM_PIXELS_OFFSET = 192
SHM_MAGIC = 0x30303456
SHM_VERSION = 1
SHM_FLAG_STATE_VALID = 1 << 0
SHM_FLAG_PIXELS_VALID = 1 << 1
HEADER = struct.Struct("<IIQQIIII")
FIELD_COUNT = 25


class CaptureError(RuntimeError):
    pass


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Harness:
    """Line-protocol client for one real ``--rl-role env`` child."""

    def __init__(self, argv: list[str], env: dict[str, str], log: list[dict]):
        self.argv = argv
        self.log = log
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        self.buffer = b""

    def ask(self, command: str, timeout: float = 120.0) -> dict[str, str]:
        record = {"command": command, "reply": None}
        self.log.append(record)
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write((command + "\n").encode("ascii"))
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CaptureError(f"timeout on {command!r}, exit={self.proc.poll()}")
            ready, _, _ = select.select([self.proc.stdout], [], [], min(remaining, 0.25))
            if ready:
                chunk = os.read(self.proc.stdout.fileno(), 8192)
                if not chunk:
                    raise CaptureError(f"pipe EOF on {command!r}, exit={self.proc.poll()}")
                self.buffer += chunk
            elif self.proc.poll() is not None:
                raise CaptureError(f"engine exited on {command!r}, exit={self.proc.returncode}")
        line, self.buffer = self.buffer.split(b"\n", 1)
        reply = line.decode("utf-8", "replace").strip()
        record["reply"] = reply
        if not reply.startswith("OK "):
            raise CaptureError(f"{command!r} rejected: {reply}")
        return dict(part.split("=", 1) for part in reply.split()[1:] if "=" in part)

    def close(self) -> dict[str, object]:
        exit_code: int | None = None
        if self.proc.poll() is None:
            try:
                self.ask("QUIT", timeout=15.0)
            except (CaptureError, OSError):
                pass
        try:
            exit_code = self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            exit_code = self.proc.wait(timeout=5)
        stderr = b""
        if self.proc.stderr is not None:
            stderr = self.proc.stderr.read() or b""
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        return {
            "argv": self.argv,
            "exit_code": exit_code,
            "stderr_tail": stderr.decode("utf-8", "replace")[-2000:],
        }


def parse_state(reply: dict[str, str]) -> dict[str, int]:
    # The harness prints f0..f5 and f13..f24 as unsigned decimals and f6..f12 as
    # %08x float bit patterns (rlharness.cpp, HandleState).  The ordering is
    # fixed, so the radix is decided by position instead of guessed from text.
    fields = {}
    for index in range(FIELD_COUNT):
        key = f"f{index}"
        if key not in reply:
            raise CaptureError(f"STATE reply missing {key}: {reply}")
        fields[key] = int(reply[key], 16) if 6 <= index <= 12 else int(reply[key], 10)
    fields["frames"] = int(reply["frames"])
    return fields


def read_shm(path: Path) -> tuple[dict[str, int], bytes]:
    raw = path.read_bytes()
    if len(raw) != SHM_PIXELS_OFFSET + PIXEL_BYTES:
        raise CaptureError(f"shared block size {len(raw)} is not {SHM_PIXELS_OFFSET + PIXEL_BYTES}")
    magic, version, frame, state_hash, width, height, state_bytes, flags = HEADER.unpack_from(raw)
    if (magic, version, width, height, state_bytes) != (
        SHM_MAGIC,
        SHM_VERSION,
        WIDTH,
        HEIGHT,
        STATE_BYTES,
    ):
        raise CaptureError(f"shared header invalid: {hex(magic)} v{version} {width}x{height}")
    if not flags & SHM_FLAG_PIXELS_VALID:
        raise CaptureError("shared block carries no pixel publication flag")
    header = {
        "magic": magic,
        "version": version,
        "frame": frame,
        "stateHash": state_hash,
        "width": width,
        "height": height,
        "stateBytes": state_bytes,
        "flags": flags,
        "stateValid": bool(flags & SHM_FLAG_STATE_VALID),
        "pixelsValid": bool(flags & SHM_FLAG_PIXELS_VALID),
        "stateSha256": sha256_bytes(raw[SHM_STATE_OFFSET:SHM_STATE_OFFSET + STATE_BYTES]),
    }
    return header, raw[SHM_PIXELS_OFFSET:]


def parse_targets(text: str | None, path: Path | None) -> list[int]:
    values: list[int] = []
    if text:
        values.extend(int(part) for part in text.split(",") if part.strip())
    if path is not None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("logicFrames", [])
        values.extend(int(item) for item in data)
    unique = sorted({value for value in values})
    if not unique:
        raise CaptureError("no target logicFrame given")
    if any(value < 0 for value in unique):
        raise CaptureError("target logicFrame must be >= 0")
    return unique


def capture(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists():
        print(f"refusing to reuse {out}; choose a new --out", file=sys.stderr)
        return 2
    if args.xvfb != "off" and not os.environ.get("V400_CAPTURE_XVFB"):
        # An inherited DISPLAY is not good enough: this WSL session exports
        # DISPLAY=:0 (WSLg) and the env role hangs before HELLO on it, while a
        # private Xvfb answers immediately.  Always take a fresh virtual display.
        os.environ["V400_CAPTURE_XVFB"] = "1"
        os.execvp("xvfb-run", ["xvfb-run", "-a", sys.executable, *sys.argv])
    if not os.environ.get("DISPLAY"):
        print("DISPLAY is not set; run under xvfb-run -a or leave --xvfb auto", file=sys.stderr)
        return 2

    binary = Path(args.binary)
    data_dir = Path(args.data_dir)
    replay = Path(args.replay)
    for required, label in ((binary, "harness binary"), (data_dir, "data dir"), (replay, "replay")):
        if not required.exists():
            print(f"missing {label}: {required}", file=sys.stderr)
            return 2
    targets = parse_targets(args.targets, Path(args.targets_file) if args.targets_file else None)

    out.mkdir(parents=True, exist_ok=False)
    raw_dir = out / "raw"
    raw_dir.mkdir()
    framework = Path(__file__).resolve().parents[2]
    git_head = None
    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(framework / "third_party/th08-rl"), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None

    shm_name = args.shm_name or f"v400wp8pair{os.getpid()}"
    shm_path = Path("/dev/shm") / shm_name
    env = dict(os.environ)
    env.update(
        {
            "TH08_AUTOPLAY_REPLAY": str(replay),
            "TH08_AUTOPLAY_STAGE": str(args.stage),
            "SDL_AUDIODRIVER": "dummy",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "LP_NUM_THREADS": "1",
            "TH08_DISABLE_CRASH_REPORTER": "1",
        }
    )
    argv = [
        str(binary),
        "--data-dir",
        str(data_dir),
        "--rl-control",
        "0,1",
        "--rl-role",
        "env",
    ]
    protocol: list[dict] = []
    frames: list[dict] = []
    process: dict[str, object] | None = None
    error: str | None = None
    client: Harness | None = None
    replay_start: dict[str, object] | None = None
    started = time.perf_counter()
    try:
        client = Harness(argv, env, protocol)
        hello = client.ask("HELLO", timeout=90.0)
        if hello.get("role") != "env" or hello.get("proto") != "1":
            raise CaptureError(f"HELLO did not confirm the env role: {hello}")
        debug = client.ask("DEBUG")
        if client.ask("READBACK 1").get("readback") != "1":
            raise CaptureError("per-frame readback unavailable")
        shm = client.ask("SHM /" + shm_name)
        if shm.get("active") != "1":
            raise CaptureError(f"shared block unavailable: {shm}")
        if int(shm.get("bytes", -1)) != SHM_PIXELS_OFFSET + PIXEL_BYTES:
            raise CaptureError(f"shared block size mismatch: {shm}")

        # TH08_AUTOPLAY_REPLAY is consumed by the title screen's Ready state, so it
        # cannot have fired before the first pumped frame.  Pump one frame at a
        # time until DEBUG reports a replay *and* STATE yields the tracked fields;
        # anything else inside the guard is a bounded, reported failure.
        state: dict[str, int] | None = None
        replay_debug: dict[str, str] = debug
        for _ in range(args.startup_guard):
            replay_debug = client.ask("DEBUG")
            if replay_debug.get("isReplay") == "1":
                reply = client.ask("STATE")
                if "f0" in reply:
                    state = parse_state(reply)
                    break
            client.ask("STEP 0 1")
        if state is None:
            raise CaptureError(
                "TH08_AUTOPLAY_REPLAY never produced a replay with tracked fields "
                f"within {args.startup_guard} frames; last DEBUG={json.dumps(replay_debug)}"
            )
        replay_start = {
            "debug": replay_debug,
            "harness_frame": int(state["f0"]),
            "replay_frame_counter": int(state["f2"]),
            "gameplay_frame_counter": int(state["f3"]),
            "stage": int(state["f1"]),
        }

        seek_guard = 0
        for target in targets:
            while state["f3"] < target:
                seek_guard += 1
                if seek_guard > 4000:
                    raise CaptureError(f"gave up seeking logicFrame {target}")
                delta = target - state["f3"]
                client.ask(f"STEP 0 {delta}")
                state = parse_state(client.ask("STATE"))
            if state["f3"] != target:
                raise CaptureError(
                    f"logicFrame {target} already passed (engine at {state['f3']}); "
                    "targets must be ordered ascending and reachable"
                )
            # HASH publishes the 25 fields into the shared block at this frame, so
            # the state bytes and the pixel bytes read next belong to one frame.
            hash_reply = client.ask("HASH")
            header, rgba = read_shm(shm_path)
            if int(header["frame"]) != state["f0"]:
                raise CaptureError(
                    f"logicFrame {target}: shared pixels belong to harness frame "
                    f"{header['frame']}, STATE reports {state['f0']}"
                )
            if int(header["stateHash"]) != int(hash_reply.get("hash", "0"), 16):
                raise CaptureError(
                    f"logicFrame {target}: shared state hash {header['stateHash']:016x} "
                    f"!= HASH reply {hash_reply.get('hash')}"
                )
            if state["f1"] != args.stage:
                raise CaptureError(f"logicFrame {target}: stage {state['f1']} != {args.stage}")
            raw_path = raw_dir / f"{target:05d}.rgba"
            raw_path.write_bytes(rgba)
            fields = {f"f{index}": int(state[f"f{index}"]) for index in range(FIELD_COUNT)}
            frames.append(
                {
                    "logicFrame": int(state["f3"]),
                    "harnessFrame": int(state["f0"]),
                    "replayFrameCounter": int(state["f2"]),
                    "stage": int(state["f1"]),
                    "rgba": str(raw_path.relative_to(out)),
                    "rgbaBytes": len(rgba),
                    "rgba_sha256": sha256_bytes(rgba),
                    "pixel_orientation": "bottom-up (OpenGL rows); convert flips to top-down",
                    "state_fields": fields,
                    "state_hash_fnv1a64": hash_reply.get("hash"),
                    "shmHeader": header,
                    "state_evidence": {
                        "algorithm": "fnv1a-64 over the 25 tracked u32 fields (linux harness)",
                        "fields": fields,
                        "fnv1a64": hash_reply.get("hash"),
                        "harness_frame": int(state["f0"]),
                        "replay_frame_counter": int(state["f2"]),
                        "stage": int(state["f1"]),
                    },
                }
            )
        status = "captured"
    except CaptureError as exc:
        error = str(exc)
        status = "blocked"
    finally:
        if client is not None:
            process = client.close()
        shm_exists = shm_path.exists()

    payload = {
        "schema": "v400-wp8-linux-capture/1",
        "status": status,
        "error": error,
        "source": "linux-harness",
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "harness_git_head": git_head,
        "data_dir": str(data_dir),
        "replay": str(replay),
        "replay_sha256": sha256_file(replay),
        "replay_stage": args.stage,
        "autoplay_env": {
            "TH08_AUTOPLAY_REPLAY": str(replay),
            "TH08_AUTOPLAY_STAGE": str(args.stage),
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "LP_NUM_THREADS": "1",
        },
        "role": "env",
        "display": os.environ.get("DISPLAY"),
        "shm_name": shm_name,
        "shm_path": str(shm_path),
        "shm_kept_after_exit": shm_exists,
        "replay_start": replay_start,
        "targets": targets,
        "framesCaptured": len(frames),
        "wallSeconds": round(time.perf_counter() - started, 3),
        "process": process,
        "protocol": protocol,
        "frames": frames,
    }
    capture_path = out / "capture.json"
    capture_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "error": error,
                "frames": len(frames),
                "logicFrames": [frame["logicFrame"] for frame in frames],
                "capture": str(capture_path),
                "process": process,
            },
            indent=2,
        )
    )
    return 0 if status == "captured" and len(frames) == len(targets) else 3


def convert(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image

    capture_file = Path(args.capture)
    payload = json.loads(capture_file.read_text(encoding="utf-8"))
    if payload.get("source") != "linux-harness" or payload.get("status") != "captured":
        print("capture.json is not a completed linux-harness capture", file=sys.stderr)
        return 3
    root = capture_file.resolve().parent
    out_dir = Path(args.out) if args.out else root
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for frame in payload["frames"]:
        raw_path = root / frame["rgba"]
        raw = raw_path.read_bytes()
        if len(raw) != PIXEL_BYTES:
            print(f"{raw_path}: {len(raw)} bytes is not {PIXEL_BYTES}", file=sys.stderr)
            return 3
        if sha256_bytes(raw) != frame["rgba_sha256"]:
            print(f"{raw_path}: raw bytes changed since capture", file=sys.stderr)
            return 3
        bottom_up = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
        top_down = np.ascontiguousarray(bottom_up[::-1])
        png_path = frames_dir / f"{frame['logicFrame']:05d}.png"
        Image.fromarray(top_down, "RGBA").save(png_path)
        with Image.open(png_path) as check:
            if check.mode != "RGBA" or check.size != (WIDTH, HEIGHT):
                print(f"{png_path}: unexpected PNG {check.mode} {check.size}", file=sys.stderr)
                return 3
            reread = np.asarray(check)
        if not np.array_equal(reread, top_down):
            print(f"{png_path}: PNG is not lossless", file=sys.stderr)
            return 3
        rows.append(
            {
                "logicFrame": int(frame["logicFrame"]),
                "harnessFrame": int(frame["harnessFrame"]),
                "replayFrameCounter": int(frame["replayFrameCounter"]),
                "stage": int(frame["stage"]),
                "image": f"frames/{png_path.name}",
                "image_sha256_rgba": sha256_bytes(top_down.tobytes()),
                "raw_rgba": frame["rgba"],
                "raw_rgba_sha256": frame["rgba_sha256"],
                "state_sha256": sha256_bytes(
                    # The digest covers game state only.  harnessFrame, the f0
                    # pump counter and the harness FNV-1a hash (which folds f0
                    # in) all shift with the title-screen offset, so they stay
                    # verbatim in the evidence but out of the digest.
                    json.dumps(
                        {
                            "algorithm": frame["state_evidence"]["algorithm"],
                            "fields": {
                                key: value
                                for key, value in frame["state_evidence"]["fields"].items()
                                if key != "f0"
                            },
                            "stage": frame["state_evidence"]["stage"],
                            "replay_frame_counter": frame["state_evidence"]["replay_frame_counter"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
                "state_evidence": frame["state_evidence"],
                "pixel_orientation": "top-down after a documented row flip of the bottom-up SHM bytes",
                "shmHeader": frame["shmHeader"],
            }
        )

    manifest = {
        "schema": "v400-wp8-frame-manifest/1",
        "source": "linux-harness",
        "role": "real th08-modern env-role capture under xvfb; replay playback through TH08_AUTOPLAY_REPLAY",
        "replay_sha256": payload["replay_sha256"],
        "replay_file": payload["replay"],
        "replay_stage": payload["replay_stage"],
        "binary": payload["binary"],
        "binary_sha256": payload["binary_sha256"],
        "harness_git_head": payload["harness_git_head"],
        "capture": {
            "display": payload["display"],
            "shm_name": payload["shm_name"],
            "engineExitCode": (payload.get("process") or {}).get("exit_code"),
            "engineExitNote": "all replies were parsed and every frame was on disk before QUIT; "
            "a non-zero exit code after QUIT is a teardown crash and is reported, not hidden",
            "pngMode": "RGBA (all four channels from the shared block, row-flipped)",
            "stateDigestAlgorithm": "sha256 of the canonical JSON state_evidence "
            "(linux-only; the harness FNV-1a 64 is preserved verbatim inside it)",
            "stateDigestExcludes": [
                "harness_frame",
                "fields.f0",
                "fnv1a64",
            ],
            "stateDigestExcludeNote": "harness_frame and fields.f0 count from process start; "
            "the harness FNV-1a hash folds f0 in, so all three are run-dependent",
        },
        "frames": rows,
    }
    manifest_path = out_dir / "linux-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "frames": len(rows),
                "logicFrames": [row["logicFrame"] for row in rows],
            },
            indent=2,
        )
    )
    return 0 if rows else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("v400-linux-frame-capture", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="run inside WSL: drive the env harness and store raw RGBA")
    cap.add_argument("--out", required=True)
    cap.add_argument("--targets", default=None, help="comma separated logicFrame values")
    cap.add_argument("--targets-file", default=None, help="JSON list or {logicFrames:[...]}")
    cap.add_argument("--binary", default="/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern")
    cap.add_argument("--data-dir", default="/root/v400-data/game")
    cap.add_argument("--replay", required=True)
    cap.add_argument("--stage", type=int, default=0)
    cap.add_argument("--shm-name", default=None)
    cap.add_argument(
        "--startup-guard",
        type=int,
        default=1200,
        help="frames to pump while waiting for the tracked fields to become defined",
    )
    cap.add_argument("--xvfb", choices=["auto", "require", "off"], default="auto")
    cap.set_defaults(func=capture)

    con = sub.add_parser("convert", help="run on Windows: raw RGBA -> RGBA PNG + manifest")
    con.add_argument("--capture", required=True)
    con.add_argument("--out", default=None)
    con.set_defaults(func=convert)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CaptureError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
