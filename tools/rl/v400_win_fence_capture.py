"""Formalized Windows frame-fence capture for the v4.0.0 WP8 paired-pixel audit.

This is the promoted version of the temporary script that produced the first
WP8 paired frame (``Document/Update/Alpha/v4.0.0/experiments/
EXP-P0-056-WP8-paired-frames-first.md``).  It keeps the same code path -- the
research fence build ``third_party/th08/build_fence/th08.exe`` renders every
frame and *holds* the game inside ``GameWindow::Render`` until this process
acknowledges the ticket, so the state read, the GDI grab and the fence
back-buffer copy provably describe one drawn frame -- and adds the three things
a multi-replay suite needs:

1. an exact seek to a starting ``logicFrame`` (a replay is played from its
   stage start, so a window deep in the stage has to be reached first),
2. an asserted stage selection (``--replay-stage`` is the engine's stage index,
   which is also what the Linux harness's ``TH08_AUTOPLAY_STAGE`` takes),
3. fail-closed guards: no residual ``th08.exe``, a hard wall-clock deadline well
   under the 180 s per-run budget, and a consecutiveness check on the captured
   ``logicFrame`` list.

What it does **not** do: it does not claim the two sides agree on game state.
The manifest it writes is a ``windows-fence`` frame manifest for
``tools/rl/v400_pixel_pair.py``; the cross-side state digests are different
algorithms by construction (see that module's docstring).

    python tools/rl/v400_win_fence_capture.py --out .tmp_x/win --run-dir .tmp_x/run \
        --replay-file th08_replay/Th8_ud0059.rpy --replay-stage 5 \
        --start-logic-frame 1200 --frames 200

Exit codes: 0 captured, 1 capture ran but did not produce a complete window,
2 refused to start (bad input / dirty environment), 3 blocked by a guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FENCE_EXE = ROOT / "third_party/th08/build_fence/th08.exe"
FENCE_ADDRESSES = ROOT / "third_party/th08/build_fence/frame-fence-addresses.json"
FENCE_MAP = ROOT / "third_party/th08/build_fence/th08.map"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def residual_game_processes() -> list[str]:
    """Names of already-running ``th08`` processes, via a tool-free probe.

    A leftover ``th08.exe`` would keep the named frame-fence section alive, so
    ``FrameFence.open`` could attach to the wrong process (the pid check catches
    it, but refusing up front is cheaper and leaves a cleaner record).
    """
    if sys.platform != "win32":
        return []
    import subprocess

    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq th08.exe", "/NH"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return []
    return [line.strip() for line in out.splitlines() if "th08.exe" in line.lower()]


def seek_to_logic_frame(
    fence,
    target: int,
    *,
    timeout_s: float,
    max_frames: int,
    bullet_threshold: int = 0,
    profile_every: int = 60,
    stop_frame: int | None = None,
) -> dict:
    """Advance the held replay to ``target`` one acknowledged hold at a time.

    Holds are configured with ``hold_every=1`` before this runs, so every
    presented frame stops inside the fence.  Acknowledging without reading the
    bulk state is far cheaper than a full sample (a few ms per frame), which is
    what makes a 1000+ frame seek fit in the per-run budget.  The telemetry block
    is updated *before* the hold, so ``logic_frame`` read here is the frame the
    game is stopped on.

    ``bullet_threshold`` implements the **pre-registered window rule**: stop at
    the first held frame at or after ``target`` whose ``active_bullets`` reaches
    the threshold.  The rule is fixed before any pixel is compared, and the whole
    observed profile is recorded, so "this window has danmaku" is a documented
    content criterion rather than a post-hoc pick.  With
    ``bullet_threshold == 0`` the seek stops at the first frame >= ``target``.
    ``stop_frame`` is the fallback cap: if the threshold is not reached by then,
    the seek stops there anyway and says so.
    """
    frames_seen = 0
    last_ticket = 0
    started = time.perf_counter()
    deadline = started + timeout_s
    profile: list[list[int]] = []
    threshold_frame: int | None = None
    max_bullets = 0
    reached_target = None
    reached_bullets = 0
    stopped_by = "frame-cap"
    while frames_seen < max_frames:
        if time.perf_counter() > deadline:
            raise TimeoutError(
                f"seek to logicFrame {target} exceeded {timeout_s:.1f}s "
                f"after {frames_seen} held frames"
            )
        # Wait for a hold we have not acknowledged yet.
        while True:
            ticket = fence.ticket
            if ticket and ticket != last_ticket and fence.hold_phase == 1:
                break
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"seek to logicFrame {target}: fence stopped holding "
                    f"(ticket {ticket}, hold_phase {fence.hold_phase})"
                )
            time.sleep(0.0002)
        telemetry = fence.telemetry()
        if telemetry.ticket != ticket:
            # The hold expired between the two reads; acknowledge and retry.
            fence.ack(telemetry.ticket)
            continue
        last_ticket = ticket
        fence.ack(ticket)
        frames_seen += 1
        logic_frame = int(telemetry.logic_frame)
        bullets = int(telemetry.active_bullets)
        max_bullets = max(max_bullets, bullets)
        if profile_every > 0 and (frames_seen == 1 or frames_seen % profile_every == 0):
            profile.append([logic_frame, bullets])
        if logic_frame >= target:
            if reached_target is None:
                reached_target = logic_frame
            if bullet_threshold <= 0:
                stopped_by = "target"
                reached_bullets = bullets
                break
            if bullets >= bullet_threshold:
                threshold_frame = logic_frame
                stopped_by = "bullet-threshold"
                reached_bullets = bullets
                break
            if stop_frame is not None and logic_frame >= stop_frame:
                stopped_by = "frame-cap"
                reached_bullets = bullets
                break
    else:
        raise TimeoutError(
            f"seek to logicFrame {target} gave up after {frames_seen} held frames "
            f"(max_frames guard)"
        )
    return {
        "logicFrame": logic_frame,
        "framesSkipped": frames_seen,
        "wallSeconds": round(time.perf_counter() - started, 3),
        "reachedTargetAt": reached_target,
        "stoppedBy": stopped_by,
        "stopFrame": stop_frame,
        "bulletThreshold": bullet_threshold,
        "bulletThresholdReachedAt": threshold_frame,
        "activeBulletsAtStop": reached_bullets,
        "maxActiveBulletsDuringSeek": max_bullets,
        "profileEveryFrames": profile_every,
        "profile": profile,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("v400-win-fence-capture", description=__doc__)
    parser.add_argument("--out", required=True, help="new output directory (frames/ + windows-manifest.json)")
    parser.add_argument("--run-dir", required=True, help="new game run directory")
    parser.add_argument("--replay-file", required=True)
    parser.add_argument(
        "--replay-stage",
        type=int,
        default=0,
        help="engine stage index (0..8); the same integer the Linux harness takes as "
        "TH08_AUTOPLAY_STAGE",
    )
    parser.add_argument("--start-logic-frame", type=int, default=0)
    parser.add_argument(
        "--seek-bullet-threshold",
        type=int,
        default=0,
        help="pre-registered window rule: stop seeking at the first frame >= "
        "--start-logic-frame whose activeBullets reaches this value (0 disables)",
    )
    parser.add_argument(
        "--max-seek-frame",
        type=int,
        default=0,
        help="give up the bullet-density rule at this logicFrame (0 = start + 2400)",
    )
    parser.add_argument("--profile-every", type=int, default=60)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--hold-every", type=int, default=1)
    parser.add_argument("--hold-max-us", type=int, default=90_000)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--wait-alive", type=float, default=20.0)
    parser.add_argument(
        "--seek-seconds",
        type=float,
        default=90.0,
        help="budget for reaching --start-logic-frame",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=170.0,
        help="hard wall-clock budget; the task requires one Windows run to stay under 180 s",
    )
    parser.add_argument("--label", default="", help="free-form label recorded in the manifest")
    args = parser.parse_args(argv)

    out = Path(args.out)
    run_dir = Path(args.run_dir)
    replay_path = Path(args.replay_file)
    if out.exists():
        print(f"refusing to reuse {out}; choose a new --out", file=sys.stderr)
        return 2
    if run_dir.exists():
        print(f"refusing to reuse {run_dir}; choose a new --run-dir", file=sys.stderr)
        return 2
    for path in (ROOT, ROOT / "tools/rl"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if not FENCE_EXE.is_file() or not FENCE_ADDRESSES.is_file() or not FENCE_MAP.is_file():
        print("missing fence build / address table / map", file=sys.stderr)
        return 2
    if not replay_path.is_file():
        print(f"missing replay {replay_path}", file=sys.stderr)
        return 2
    if args.frames < 1 or args.start_logic_frame < 0:
        print("--frames must be positive and --start-logic-frame >= 0", file=sys.stderr)
        return 2
    running = residual_game_processes()
    if running:
        print(f"refusing to start: th08 already running: {running}", file=sys.stderr)
        return 3

    import os

    os.environ["TH08_ADDRESS_FILE"] = str(FENCE_ADDRESSES)

    import numpy as np
    from PIL import Image

    from frame_fence_round1 import prepare_fence_template, text_section_sha256

    from touhou_rl import menu, spec
    from touhou_rl.framefence import FenceFrameSampler, FrameFence
    from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir
    from touhou_rl.v380.fence import FencePixelSource
    from touhou_rl.v380.provenance import REQUIRED_CHECKS
    from touhou_rl.v400.gym.replay_starts import decode_replay

    # The stage index must exist in this replay, otherwise the menu would land on
    # the next present stage and the run would be mislabelled.
    try:
        stages = [start.stage_index for start in decode_replay(replay_path)]
    except Exception as exc:  # noqa: BLE001 - reported verbatim
        print(f"cannot decode replay {replay_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.replay_stage not in stages:
        print(
            f"stage {args.replay_stage} is not present in {replay_path.name}; "
            f"present stages: {stages}",
            file=sys.stderr,
        )
        return 2

    out.mkdir(parents=True, exist_ok=False)
    frames_dir = out / "frames"
    frames_dir.mkdir()
    build_identity = {
        "exeSha256": sha256_file(FENCE_EXE),
        "textSha256": text_section_sha256(FENCE_EXE),
        "addressTableSha256": sha256_file(FENCE_ADDRESSES),
        "mapSha256": sha256_file(FENCE_MAP),
        "exeBytes": FENCE_EXE.stat().st_size,
    }
    environment = {
        "platform": sys.platform,
        "gpu": "RTX 4070 SUPER / driver 617.42",
        "driver": "617.42",
        "python": sys.version.split()[0],
        "windowMode": "windowed 640x480, D3D8 COPY, no vsync",
        "clientSize": [640, 480],
        "pixelSource": "GDI grab of the held window, verified equal to the fence backbuffer copy",
        "inputBackend": "SendInput/PostMessage via GameSession",
    }

    template = prepare_fence_template()
    prepare_run_dir(
        run_dir,
        template,
        windowed=True,
        default_difficulty=spec.NORMAL,
        use_postmessage_input=False,
        force=True,
    )

    session = GameSession(SessionOptions(run_dir=run_dir, windowed=True, boot_timeout_s=45.0))
    fence = None
    sampler = None
    source = None
    rows: list[dict] = []
    failures: list[str] = []
    error: str | None = None
    seek_info: dict | None = None
    observed_stage: int | None = None
    started = time.perf_counter()
    deadline = started + args.deadline_seconds
    final_fence: dict = {}
    try:
        session.start()
        fence = FrameFence.open(timeout=20.0)
        if fence.header()["pid"] != session.pid:
            raise RuntimeError("stale fence section: pid mismatch")
        fence.configure(enable=True, hold_every=0, hold_max_us=args.hold_max_us)

        menu.install_replay(run_dir, replay_path, index=1)
        menu.enter_replay(session, replay_index=0, stage=args.replay_stage)
        try:
            session.keyboard.release_all()
        except Exception:  # noqa: BLE001
            pass

        stage_deadline = time.perf_counter() + args.wait_alive
        while time.perf_counter() < stage_deadline:
            current = session.read_state(with_bullets=False)
            observed_stage = int(current.stage)
            if observed_stage == args.replay_stage and int(current.player_state) == 0 and not current.is_demo:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                f"replay did not reach stage {args.replay_stage} with a live player "
                f"(observed stage {observed_stage})"
            )

        warmup_deadline = time.perf_counter() + (0.0 if args.start_logic_frame > 0 else args.warmup_seconds)
        while time.perf_counter() < warmup_deadline:
            time.sleep(0.05)

        # Bootstrap: the fence only honours holds after a non-zero ack.  Holds go
        # on *before* the seek so the replay cannot free-run past the requested
        # window (the engine is 60fps-capped, so a warmup would overshoot a small
        # target).
        start_ticket = fence.ticket
        wait_until = time.perf_counter() + 5.0
        while fence.ticket == start_ticket and time.perf_counter() < wait_until:
            time.sleep(0.0005)
        fence.ack(fence.ticket)
        fence.configure(
            enable=True, pixels=False, hold_every=1, hold_min_us=0, hold_max_us=args.hold_max_us
        )
        if args.start_logic_frame > 0:
            max_seek = args.max_seek_frame or (args.start_logic_frame + 2400)
            seek_info = seek_to_logic_frame(
                fence,
                args.start_logic_frame,
                timeout_s=min(args.seek_seconds, max(1.0, deadline - time.perf_counter())),
                max_frames=max(4096, max_seek * 2 + 1024),
                bullet_threshold=args.seek_bullet_threshold,
                profile_every=args.profile_every,
                stop_frame=max_seek if args.seek_bullet_threshold > 0 else None,
            )
            if (
                args.seek_bullet_threshold > 0
                and seek_info["bulletThresholdReachedAt"] is None
            ):
                seek_info["bulletThresholdNote"] = (
                    f"no frame at or before {max_seek} reached "
                    f"{args.seek_bullet_threshold} active bullets; the window starts at "
                    f"{seek_info['logicFrame']} with {seek_info['activeBulletsAtStop']} bullets"
                )
        fence.configure(
            enable=True, pixels=True, hold_every=args.hold_every, hold_min_us=0,
            hold_max_us=args.hold_max_us,
        )
        sampler = FenceFrameSampler(
            session, fence, with_pixels=True, with_enemies=True, accept_current=True
        )
        source = FencePixelSource(sampler, timeout=3.0)

        for index in range(args.frames):
            if time.perf_counter() > deadline:
                failures.append(
                    f"deadline: stopped before frame index {index} "
                    f"({args.deadline_seconds:.0f}s budget)"
                )
                break
            sample = sampler.next(timeout=3.0)
            if sample is None:
                failures.append(f"index {index}: no fence sample within timeout")
                continue
            try:
                observation = source.accept(sample)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"index {index}: {type(exc).__name__}: {exc}")
                continue
            telemetry = sample.telemetry
            state = sample.state
            logic_frame = int(observation.frame)
            rgb = np.ascontiguousarray(observation.pixels[:, :, :3])
            rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
            png_path = frames_dir / f"{logic_frame:05d}.png"
            Image.fromarray(rgba, "RGBA").save(png_path)
            evidence = {
                "bullet_digest": f"{int(telemetry.bullet_digest):08x}",
                "enemy_digest": f"{int(telemetry.enemy_digest):08x}",
                "digest_ticket": int(telemetry.digest_ticket),
                "active_bullets": int(telemetry.active_bullets),
                "player_bits": list(telemetry.player_bits()),
                "player_x": float(state.player_x),
                "player_y": float(state.player_y),
                "player_vx": float(state.player_vx),
                "player_vy": float(state.player_vy),
                "player_state": int(state.player_state),
                "focus_mode": int(state.focus_mode),
                "input_mask": int(state.input_mask),
                "gameplay_frame": int(state.gameplay_frame),
                "stage_frames": int(state.stage_frames),
                "score": int(state.score),
                "lives": float(state.lives),
                "power": float(state.power),
                "is_replay": bool(state.is_replay),
                "stage": int(state.stage),
            }
            checks = {name: bool(sample.checks.get(name, False)) for name in REQUIRED_CHECKS}
            # The digest covers game state only: the capture ticket is real
            # evidence, but it changes every run and would make the "state"
            # digest run-dependent.
            digest_input = {key: value for key, value in evidence.items() if key != "digest_ticket"}
            state_sha = sha256_bytes(
                json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            rows.append(
                {
                    "logicFrame": logic_frame,
                    "ticket": int(sample.ticket),
                    "image": f"frames/{png_path.name}",
                    "image_sha256_rgba": sha256_bytes(rgba.tobytes()),
                    "image_sha256_rgb": sha256_bytes(rgb.tobytes()),
                    "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
                    "state_sha256": state_sha,
                    "state_evidence": evidence,
                    "checks": checks,
                    "checks_all_pass": all(checks.values()) and len(checks) == len(REQUIRED_CHECKS),
                    "hold_us": int(telemetry.hold_last_us),
                    "note": observation.note,
                }
            )
        final_fence = {
            "ticket": int(fence.ticket),
            "holdCount": int(fence.telemetry().hold_count),
            "holdBudgetSkips": int(fence.telemetry().hold_budget_skips),
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        error = f"{type(exc).__name__}: {exc}"
        final_fence = {"error": error, "traceback": traceback.format_exc()}
    finally:
        if sampler is not None:
            final_fence = {**final_fence, "sampler": {
                "samples": int(sampler.samples),
                "retries": int(sampler.retries),
                "attempts": int(sampler.attempts),
                "lastFailure": sampler.last_failure,
            }}
        if fence is not None:
            try:
                fence.disable()
            except Exception:  # noqa: BLE001
                pass
            fence.close()
        session.close()

    logic_frames = [row["logicFrame"] for row in rows]
    consecutive = bool(logic_frames) and logic_frames == list(
        range(logic_frames[0], logic_frames[0] + len(logic_frames))
    )
    manifest = {
        "schema": "v400-wp8-frame-manifest/1",
        "source": "windows-fence",
        "role": "research fence build capture; not a retail-original capture",
        "label": args.label,
        "replay_sha256": sha256_file(replay_path),
        "replay_file": str(replay_path),
        "replay_stage": args.replay_stage,
        "replay_stages_present": stages,
        "build": build_identity,
        "environment": environment,
        "capture": {
            "mode": "replay",
            "holdEvery": args.hold_every,
            "framesRequested": args.frames,
            "framesCaptured": len(rows),
            "startLogicFrameRequested": args.start_logic_frame,
            "firstLogicFrame": logic_frames[0] if logic_frames else None,
            "lastLogicFrame": logic_frames[-1] if logic_frames else None,
            "logicFramesConsecutive": consecutive,
            "seek": seek_info,
            "observedStage": observed_stage,
            "wallSeconds": round(time.perf_counter() - started, 3),
            "deadlineSeconds": args.deadline_seconds,
            "failures": failures,
            "error": error,
            "fenceFinal": final_fence,
            "pngMode": "RGBA",
            "pngAlpha": "synthetic constant 255; the fence pixel source is RGB24",
            "stateDigestAlgorithm": "sha256 of the canonical JSON state_evidence "
            "(windows-only; not comparable with the linux 25-field FNV-1a hash)",
            "stateDigestExcludes": ["digest_ticket"],
        },
        "frames": rows,
    }
    manifest_path = out / "windows-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "manifest": str(manifest_path),
        "stage": args.replay_stage,
        "frames": len(rows),
        "firstLogicFrame": manifest["capture"]["firstLogicFrame"],
        "lastLogicFrame": manifest["capture"]["lastLogicFrame"],
        "consecutive": consecutive,
        "allChecksPass": all(row["checks_all_pass"] for row in rows) if rows else False,
        "wallSeconds": manifest["capture"]["wallSeconds"],
        "failures": failures,
        "error": error,
    }
    print(json.dumps(summary, indent=2))
    ok = bool(rows) and not failures and error is None and consecutive and len(rows) == args.frames
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
