"""Capture a small dataset of *provably same-frame* TH08 samples.

Every sample is produced by the frame-fence protocol (see
``touhou_rl/framefence.py`` and ``third_party/th08/docs/FRAME_FENCE.md``): the
game holds inside the fence, the reader captures the client area and reads the
state, both are validated against the digests and the back-buffer copy the game
published for that ticket, and only then is the ticket acknowledged.  A sample
whose hold ended mid-read is discarded, never written.

    python tools/rl/frame_fence_capture.py --frames 120 --out-dir .tmp/exact-frames
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FENCE_ADDRESSES = ROOT / "third_party/th08/build_fence/frame-fence-addresses.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("frame-fence-capture")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out-dir", default=".tmp/exact-frames")
    parser.add_argument("--run-dir", default=".tmp/exact-frames-run")
    parser.add_argument("--mode", choices=["demo", "practice"], default="demo")
    parser.add_argument(
        "--hold-every",
        type=int,
        default=2,
        help="hold every n-th drawn frame; 1 holds every frame but then the slow "
        "PNG write in this tool makes the game burn its hold budget and drop frames",
    )
    parser.add_argument("--hold-max-us", type=int, default=90_000)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    out_dir = (ROOT / args.out_dir).resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TH08_ADDRESS_FILE"] = str(FENCE_ADDRESSES)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools/rl"))

    from frame_fence_round1 import prepare_fence_template, text_section_sha256
    from touhou_rl import menu, spec
    from touhou_rl.framefence import FenceFrameSampler, FrameFence
    from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir

    template = prepare_fence_template()
    run_dir = (ROOT / args.run_dir).resolve()
    prepare_run_dir(
        run_dir,
        template,
        windowed=True,
        default_difficulty=spec.LUNATIC,
        use_postmessage_input=True,
        force=True,
    )

    manifest = {
        "schema": "th08-fence-capture/1",
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "requestedFrames": args.frames,
        "mode": args.mode,
        "holdEvery": args.hold_every,
        "imageSource": "gdi-client-area (validated byte-equal to the published back buffer)",
        "build": {
            "exeSha256": _sha256(ROOT / "third_party/th08/build_fence/th08.exe"),
            "textSha256": text_section_sha256(ROOT / "third_party/th08/build_fence/th08.exe"),
            "addressFileSha256": _sha256(FENCE_ADDRESSES),
        },
        "addressFile": "third_party/th08/build_fence/frame-fence-addresses.json",
        "note": (
            "Produced by the research-only fence build; NOT the canonical image. "
            "Every record passed: hold_stable, player position/velocity/state/focus, "
            "gameplay frame, stage frames, input mask, active bullets, bullet digest, "
            "enemy digest, pixel freeze, pixels==back buffer."
        ),
    }

    session = GameSession(SessionOptions(run_dir=run_dir, windowed=True, boot_timeout_s=45.0))
    fence = None
    written = 0
    failures = 0
    checks_total = 0
    started = time.perf_counter()
    jsonl_path = out_dir / "capture.jsonl"
    try:
        session.start()
        fence = FrameFence.open(timeout=20.0)
        if fence.header()["pid"] != session.pid:
            raise SystemExit("stale fence section: pid mismatch")
        fence.configure(enable=True, hold_every=0, hold_max_us=args.hold_max_us)

        if args.mode == "practice":
            menu.enter_game(session, difficulty=spec.LUNATIC, practice_stage=0)
        else:
            deadline = time.perf_counter() + 30.0
            while time.perf_counter() < deadline:
                if session.read_state(with_bullets=False).supervisor_state == spec.STATE_GAME_MANAGER:
                    break
                time.sleep(0.05)
            else:
                raise SystemExit("attract demo never started")

        deadline = time.perf_counter() + args.warmup_seconds
        while time.perf_counter() < deadline:
            time.sleep(0.05)

        # Acknowledge one ordinary ticket so the game starts honouring holds.
        start_ticket = fence.ticket
        while fence.ticket == start_ticket:
            time.sleep(0.0005)
        fence.ack(fence.ticket)
        fence.configure(
            enable=True,
            pixels=True,
            hold_every=args.hold_every,
            hold_min_us=0,
            hold_max_us=args.hold_max_us,
        )
        sampler = FenceFrameSampler(session, fence, with_pixels=True, with_enemies=True)

        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            for index in range(args.frames):
                sample = sampler.next(timeout=3.0, attempts=12)
                if sample is None:
                    failures += 1
                    continue
                telemetry = sample.telemetry
                state = sample.state
                assert telemetry is not None and state is not None
                checks_total += len(sample.checks)
                if not sample.ok:
                    failures += 1
                    continue
                assert sample.pixels is not None
                image_name = f"{written:05d}-ticket-{sample.ticket:06d}.png"
                try:
                    from PIL import Image

                    Image.fromarray(sample.pixels).save(frames_dir / image_name, compress_level=1)
                except Exception as exc:  # pragma: no cover - diagnostics only
                    failures += 1
                    manifest.setdefault("imageErrors", []).append(repr(exc))
                    continue
                record = {
                    "index": written,
                    "ticket": sample.ticket,
                    "logicFrame": telemetry.logic_frame,
                    "stageFrames": telemetry.stage_frames,
                    "runFrames": telemetry.run_frames,
                    "stage": int(state.stage),
                    "isDemo": bool(state.is_demo),
                    "isReplay": bool(state.is_replay),
                    "isPractice": bool(state.is_practice),
                    "player": {
                        "x": telemetry.player_x,
                        "y": telemetry.player_y,
                        "vx": telemetry.player_vx,
                        "vy": telemetry.player_vy,
                        "state": int(telemetry.player_state),
                        "focus": int(telemetry.focus_mode),
                    },
                    "inputMask": int(telemetry.input_mask),
                    "activeBullets": int(telemetry.active_bullets),
                    "bulletDigest": int(telemetry.bullet_digest),
                    "enemyDigest": int(telemetry.enemy_digest),
                    # `hold_last_us` is the hold that had just finished when the
                    # telemetry was read (the game updates it after ack), so it
                    # is reported as such.  The reader's own work inside the
                    # hold is the accurate per-sample cost.
                    "gameHoldPreviousUs": int(telemetry.hold_last_us),
                    "readerWorkUs": round(sample.timings_us["total"], 1),
                    "image": f"frames/{image_name}",
                    "imageSha256": hashlib.sha256(sample.pixels.tobytes()).hexdigest().upper(),
                    "checks": sample.checks,
                }
                jsonl.write(json.dumps(record, sort_keys=True) + "\n")
                written += 1

        manifest["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        manifest["wallSeconds"] = round(time.perf_counter() - started, 3)
        manifest["samples"] = written
        manifest["discarded"] = failures
        manifest["checksPassed"] = checks_total
        manifest["fenceFinal"] = {
            "ticket": fence.ticket,
            "holdCount": fence.telemetry().hold_count,
            "holdMaxUs": fence.telemetry().hold_max_us,
            "holdBudgetSkips": fence.telemetry().hold_budget_skips,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 0 if written else 1
    finally:
        if fence is not None:
            try:
                fence.disable()
            except Exception:
                pass
            fence.close()
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
