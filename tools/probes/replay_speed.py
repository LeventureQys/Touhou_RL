"""Measure whether TH08 replay playback can be fast-forwarded (and by how much).

Why this matters
----------------
Collection is bound by *playback* speed: a Lunatic full run is ~150k logic frames
of 60 Hz game time, i.e. ~42 minutes of wall clock per file, and the project has
48 replays (~21 h).  Every speedup factor translates directly into hours saved, so
before writing a batch driver it is worth asking the game itself:

* does holding the "skip" key (default Ctrl) during replay playback make the logic
  frame counter advance faster than 60 Hz?
* if so, at what rate, and does the counter stay monotonic (no skipped frames)?

The probe measures logic frames per wall-clock second in three phases: idle,
holding Ctrl, idle again.  Nothing is written to the game.
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
RUN_DIR = ROOT / "third_party/th08/build/rl/replay-speed"


def measure_phase(session, seconds: float, hold_ctrl: bool) -> dict:
    """Return logic frames advanced per second while sampling as fast as possible."""
    reader = session.reader
    assert reader is not None
    if hold_ctrl:
        session.keyboard.set_keys({"ctrl"})
    else:
        session.keyboard.release_all()
    time.sleep(0.05)
    start = reader.read_state(with_bullets=False, with_title=False)
    frames = 0
    samples = 0
    last = start.gameplay_frame
    regressions = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        state = reader.read_state(with_bullets=False, with_title=False)
        if state.supervisor_state == spec.STATE_GAME_MANAGER:
            delta = (state.gameplay_frame - last) & 0xFFFFFFFF
            if delta:
                frames += delta
                samples += 1
                if delta > 0x80000000:  # wrapped backwards: replay looped / restarted
                    regressions += 1
                last = state.gameplay_frame
    session.keyboard.release_all()
    elapsed = seconds
    return {
        "seconds": elapsed,
        "logic_frames": frames,
        "logic_fps": round(frames / elapsed, 1),
        "samples": samples,
        "regressions": regressions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure TH08 replay fast-forward capability")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument(
        "--out", default=str(ROOT / "third_party/th08/build/rl/probe-v103/replay-speed.json")
    )
    args = parser.parse_args(argv)

    prepare_run_dir(RUN_DIR, TEMPLATE, windowed=True)
    menu.install_replay(RUN_DIR, Path(args.replay))
    session = GameSession(SessionOptions(run_dir=RUN_DIR))
    report: dict = {"replay": str(args.replay), "phases": {}}
    try:
        session.start()
        stage = 0
        try:
            from touhou_rl.capture_dataset import replay_first_stage

            stage = replay_first_stage(Path(args.replay))
        except Exception:
            pass
        menu.enter_replay(session, stage=stage, timeout=60.0)
        report["phases"]["idle_before"] = measure_phase(session, args.seconds, hold_ctrl=False)
        report["phases"]["ctrl_held"] = measure_phase(session, args.seconds, hold_ctrl=True)
        report["phases"]["idle_after"] = measure_phase(session, args.seconds, hold_ctrl=False)
        base = report["phases"]["idle_before"]["logic_fps"] or 1.0
        fast = report["phases"]["ctrl_held"]["logic_fps"]
        report["speedup"] = round(fast / base, 2)
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
