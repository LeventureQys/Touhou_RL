"""Break down the per-sample cost of vision capture, stage by stage.

Why: "16.6 samples/s" is a system number.  The only way to know whether it is a
fundamental limit (PNG encoding) or a fixable one (grabbing the window, writing to
disk, the flush) is to time every stage separately during a real playback.

Reported per sample: grab / read_state / read_bullets / crops / PNG per region /
writer.add, plus the amortised shard flush.  The implied ceiling
``1000 / total_ms`` is compared against ``60 / every`` (what the game can offer),
which is exactly the reasoning behind the chosen sampling rate.

Usage::

    python tools/probes/capture_profile.py --replay th08_replay/Th8_ud3936.rpy --seconds 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from touhou_rl import menu, obs, spec, vision, winapi
from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir
from touhou_rl.vision_data import VisionShardWriter

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "third_party/th08/build/play"
RUN_DIR = ROOT / "third_party/th08/build/rl/profile-session"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile per-stage cost of vision capture")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--compress-level", type=int, default=6)
    parser.add_argument("--out", default=str(ROOT / "third_party/th08/build/rl/probe-v103/capture-profile.json"))
    args = parser.parse_args(argv)

    prepare_run_dir(RUN_DIR, TEMPLATE, windowed=True)
    menu.install_replay(RUN_DIR, Path(args.replay))
    session = GameSession(SessionOptions(run_dir=RUN_DIR))
    out_dir = ROOT / "third_party/th08/build/rl/datasets/_profile"
    writer = VisionShardWriter(out_dir, shard_size=args.shard_size, tag="profile", allow_append=True)
    timing: dict[str, float] = {}
    counts: dict[str, int] = {}
    samples = 0

    def add(name: str, seconds: float) -> None:
        timing[name] = timing.get(name, 0.0) + seconds
        counts[name] = counts.get(name, 0) + 1

    report: dict = {"every": args.every, "seconds": args.seconds}
    started = time.perf_counter()
    try:
        stage = 0
        try:
            from touhou_rl.capture_dataset import replay_first_stage

            stage = replay_first_stage(Path(args.replay))
        except Exception:
            pass
        session.start()
        menu.enter_replay(session, stage=stage, timeout=60.0)
        reader = session.reader
        assert reader is not None
        session.keyboard.release_all()
        last_key: tuple[int, int] | None = None
        next_target: int | None = None
        last_guard = 0.0
        while time.perf_counter() - started < args.seconds:
            now = time.perf_counter()
            if now - last_guard > 2.0:
                last_guard = now
                if winapi.user32.GetForegroundWindow() != session.hwnd:
                    winapi.force_foreground(session.hwnd)
            t0 = time.perf_counter()
            state = reader.read_state(with_bullets=True, with_title=False)
            add("read_state", time.perf_counter() - t0)
            if state.supervisor_state != spec.STATE_GAME_MANAGER:
                continue
            key = (state.stage, state.gameplay_frame)
            if last_key is not None:
                if key == last_key:
                    continue
                if key[0] != last_key[0]:
                    next_target = None  # stage boundary resets the stride clock
            # Absolute-target stride (same rule as vision_capture): ``delta % every``
            # drops frames whenever a sample lands a few ms late.
            if next_target is not None and key[1] < next_target:
                continue
            next_target = key[1] + args.every
            last_key = key

            t0 = time.perf_counter()
            bullets = reader.read_bullets()
            add("read_bullets", time.perf_counter() - t0)
            t0 = time.perf_counter()
            observation = obs.encode(state, bullets)
            add("obs_encode", time.perf_counter() - t0)
            t0 = time.perf_counter()
            screen = session.grab()
            add("grab", time.perf_counter() - t0)
            t0 = time.perf_counter()
            crops = vision.extract_regions(screen)
            add("crops", time.perf_counter() - t0)
            blobs = {}
            for name, image in crops.items():
                t0 = time.perf_counter()
                blobs[name] = vision.encode_png(image, compress_level=args.compress_level)
                add(f"png_{name}", time.perf_counter() - t0)
            t0 = time.perf_counter()
            writer.add(
                blobs,
                observation,
                spec.buttons_to_action(state.gui_input_current),
                {
                    "frame": state.gameplay_frame,
                    "stage": state.stage,
                    "score": state.score,
                    "graze": state.graze,
                    "lives": state.lives,
                    "bombs": state.bombs,
                    "power": state.power,
                    "deaths": state.deaths,
                },
                native_hw={name: (image.shape[0], image.shape[1]) for name, image in crops.items()},
            )
            add("writer_add", time.perf_counter() - t0)
            samples += 1
        elapsed = time.perf_counter() - started
        t0 = time.perf_counter()
        summary = writer.close()
        flush_seconds = time.perf_counter() - t0
    finally:
        try:
            session.close()
        except Exception:
            pass

    per_sample = {name: round(1000.0 * seconds / max(1, counts.get(name, 1)), 2) for name, seconds in timing.items()}
    total_ms = round(1000.0 * elapsed / max(1, samples), 2)
    breakdown = {
        "samples": samples,
        "elapsed_seconds": round(elapsed, 2),
        "achieved_samples_per_s": round(samples / elapsed, 2),
        "total_ms_per_sample": total_ms,
        "stage_ms_per_sample": per_sample,
        "flush_amortised_ms_per_sample": round(1000.0 * flush_seconds / max(1, samples), 2),
        "implied_ceiling_samples_per_s": round(1000.0 / max(0.001, total_ms), 2),
        "game_supply_at_every": round(60.0 / args.every, 2),
        "bytes": summary.get("frame_bytes", 0),
        "bytes_per_sample": round(summary.get("frame_bytes", 0) / max(1, samples), 0),
    }
    report.update(breakdown)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
