"""Does grabbing the game window slow the game itself?

Why: the per-stage profile shows only ~22 ms of work per sample, yet the achieved
rate is far below ``60 / every``.  Two candidate explanations remain:

1. the sampling loop is late (scheduling), or
2. ``BitBlt`` from the D3D window surface steals time from the game's own loop, so
   the *supply* of logic frames drops.

This probe separates them: it measures replay playback frame rate in three phases -
no grabbing, grabbing at ~30 Hz, no grabbing - and prints the implied ceilings.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from touhou_rl import menu, spec, winapi
from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "third_party/th08/build/play"
RUN_DIR = ROOT / "third_party/th08/build/rl/grab-impact"


def phase(session, seconds: float, grab: bool, grab_interval: float = 1 / 30) -> dict:
    reader = session.reader
    assert reader is not None
    start = reader.read_state(with_bullets=False, with_title=False)
    last_frame = start.gameplay_frame
    frames = 0
    grabs = 0
    next_grab = time.perf_counter()
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        state = reader.read_state(with_bullets=False, with_title=False)
        if state.supervisor_state == spec.STATE_GAME_MANAGER:
            delta = (state.gameplay_frame - last_frame) & 0xFFFFFFFF
            if delta:
                frames += delta
                last_frame = state.gameplay_frame
        if grab and time.perf_counter() >= next_grab:
            session.grab()
            grabs += 1
            next_grab = time.perf_counter() + grab_interval
    return {
        "seconds": seconds,
        "logic_frames": frames,
        "logic_fps": round(frames / seconds, 1),
        "grabs": grabs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the game-loop cost of window grabbing")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--out", default=str(ROOT / "third_party/th08/build/rl/probe-v103/grab-impact.json"))
    args = parser.parse_args(argv)

    prepare_run_dir(RUN_DIR, TEMPLATE, windowed=True)
    menu.install_replay(RUN_DIR, Path(args.replay))
    session = GameSession(SessionOptions(run_dir=RUN_DIR))
    report: dict = {"replay": str(args.replay), "phases": {}}
    try:
        stage = 0
        try:
            from touhou_rl.capture_dataset import replay_first_stage

            stage = replay_first_stage(Path(args.replay))
        except Exception:
            pass
        session.start()
        menu.enter_replay(session, stage=stage, timeout=60.0)
        session.keyboard.release_all()
        report["phases"]["no_grab_before"] = phase(session, args.seconds, grab=False)
        report["phases"]["grab_30hz"] = phase(session, args.seconds, grab=True)
        report["phases"]["no_grab_after"] = phase(session, args.seconds, grab=False)
        base = report["phases"]["no_grab_before"]["logic_fps"] or 1.0
        slowed = report["phases"]["grab_30hz"]["logic_fps"]
        report["fps_ratio"] = round(slowed / base, 3)
        report["ceiling_at_every_2"] = round(slowed / 2, 2)
        report["ceiling_at_every_3"] = round(slowed / 3, 2)
        report["ceiling_at_every_4"] = round(slowed / 4, 2)
    finally:
        try:
            session.close()
        except Exception:
            pass
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
