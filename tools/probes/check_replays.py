"""Check that a replay actually plays back on this build (usability smoke test).

Offline validation (``third_party/th08/scripts/replay_file.py``) proves that a
replay file is structurally intact, checksummed, and recorded by retail TH08
1.00d.  It does *not* prove the rebuilt binary can play it: the replay menu, the
input feed through ``g_GuiMessageInputCurrent`` and the stage hand-off are all
runtime behaviour.

This probe closes that gap.  For every given file it installs the replay into a
throwaway run directory, walks the title screen into the replay player, samples
a few seconds, and reports:

* whether replay mode actually engaged (``is_replay`` + ``replay_mode``),
* how many gameplay frames advanced (a stalled counter means a broken load),
* how many sampled frames carried a non-zero expert input word - this is exactly
  the channel ``touhou_rl.capture_dataset`` reads expert actions from, so 0 here
  means "the file plays but cannot be learned from",
* the stage the replay started on.

Everything is read-only; nothing is written to the game.

Usage::

    python tools/probes/check_replays.py --replay th08_replay/Th8_ud3936.rpy --seconds 8
    python tools/probes/check_replays.py --dir th08_replay --seconds 5 --out probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from touhou_rl import menu, spec, winapi
from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "third_party/th08/build/play"
RUN_DIR = ROOT / "third_party/th08/build/rl/replay-check"
# The replay corpus validator lives with the reconstruction sources; reuse it
# instead of re-deriving the header layout here.
_REPLAY_SCRIPTS = ROOT / "third_party/th08/scripts"
if str(_REPLAY_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_REPLAY_SCRIPTS))


def first_stage(replay: Path) -> int:
    """Stage index the replay starts on (Extra replays only offer stage 8).

    Hardcoding 0 breaks Extra: its replay has a single entry, the Extra stage
    (index 8), so the stage cursor can never reach 0 and the menu walk times out.
    """
    try:
        from replay_file import inspect

        return int(inspect(Path(replay))["stages"][0]["index"])
    except Exception:
        return 0


def check_one(replay: Path, seconds: float) -> dict:
    prepare_run_dir(RUN_DIR, TEMPLATE, windowed=True)
    menu.install_replay(RUN_DIR, Path(replay))
    session = GameSession(SessionOptions(run_dir=RUN_DIR))
    row = {"replay": str(replay), "name": Path(replay).name, "ok": False, "reason": ""}
    started = time.time()
    stage = first_stage(Path(replay))
    row["start_stage_requested"] = stage
    try:
        session.start()
        try:
            menu.enter_replay(session, stage=stage, timeout=60.0)
        except Exception as exc:
            # Diagnose instead of just reporting a timeout: a replay the game
            # refuses shows up as "menu never reached the replay screen", and the
            # difference between "the game rejected the file" and "the game died"
            # matters for what to do next.
            try:
                reader = session.reader
                state = reader.read_state(with_bullets=False, with_title=True) if reader else None
                row["diagnostics"] = {
                    "process_alive": session.process is not None and session.process.poll() is None,
                    "supervisor_state": getattr(state, "supervisor_state", None),
                    "title_screen_current": getattr(state, "title_screen_current", None),
                    "title_screen_state": getattr(state, "title_screen_state", None),
                    "title_cursor": getattr(state, "title_cursor", None),
                    "title_replay_count": getattr(state, "title_replay_count", None),
                }
            except Exception as inner:
                row["diagnostics"] = {"diagnostics_failed": repr(inner)}
            row["reason"] = f"{type(exc).__name__}: {exc}"
            return row
        reader = session.reader
        assert reader is not None
        session.keyboard.release_all()
        first = reader.read_state(with_bullets=False, with_title=False)
        row["start_stage"] = first.stage
        row["start_frame"] = first.gameplay_frame
        row["is_replay"] = bool(first.is_replay)
        row["replay_mode"] = first.replay_mode
        frames = 0
        sampled = 0
        nonzero = 0
        last_guard = 0.0
        last_frame = first.gameplay_frame
        playback_started: float | None = None
        # The sampling budget starts when playback actually starts.  Measuring it
        # from process launch (as the first version did) leaves ~0 s of window
        # whenever the boot + menu walk is slow, which silently reports "no expert
        # input observed" for perfectly good replays.
        hard_deadline = started + seconds + 90.0
        while time.time() < hard_deadline:
            now = time.time()
            if now - last_guard > 2.0:
                last_guard = now
                if winapi.user32.GetForegroundWindow() != session.hwnd:
                    winapi.force_foreground(session.hwnd)
            state = reader.read_state(with_bullets=False, with_title=False)
            if state.supervisor_state == spec.STATE_GAME_MANAGER:
                advanced = (state.gameplay_frame - last_frame) & 0xFFFFFFFF
                if advanced:
                    if playback_started is None:
                        playback_started = now
                    frames += advanced
                    last_frame = state.gameplay_frame
                    sampled += 1
                    if state.gui_input_current:
                        nonzero += 1
            if playback_started is not None and now - playback_started >= seconds:
                break
            time.sleep(0.002)
        row["playback_wait_seconds"] = round((playback_started or hard_deadline) - started, 1)
        row.update(
            {
                "ok": frames > 0,
                "frames_advanced": frames,
                "sampled_frames": sampled,
                "nonzero_input_frames": nonzero,
                "input_coverage": round(nonzero / sampled, 3) if sampled else 0.0,
                "end_stage": state.stage,
                "end_frame": state.gameplay_frame,
                "elapsed_seconds": round(time.time() - started, 1),
            }
        )
        if frames == 0:
            row["reason"] = "gameplay frame counter did not advance"
        elif nonzero == 0:
            row["reason"] = "no expert input observed (playable but not learnable)"
    except Exception as exc:  # a replay the build cannot load usually fails here
        row["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            session.close()
        except Exception:
            pass
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Playback smoke test for TH08 replays")
    parser.add_argument("--replay", action="append", default=[], help="replay file; repeat for several")
    parser.add_argument("--dir", default="", help="check every .rpy in this directory")
    parser.add_argument("--seconds", type=float, default=6.0, help="seconds of gameplay to sample")
    parser.add_argument("--attempts", type=int, default=3, help="launch attempts per replay")
    parser.add_argument(
        "--out",
        default=str(ROOT / "third_party/th08/build/rl/probe-v103/replay-check.json"),
    )
    args = parser.parse_args(argv)
    targets = [Path(p) for p in args.replay]
    if args.dir:
        targets += sorted(Path(args.dir).glob("*.rpy"))
    if not targets:
        parser.error("give --replay FILE and/or --dir DIR")

    rows = []
    for replay in targets:
        row = None
        for attempt in range(1, args.attempts + 1):
            row = check_one(replay, args.seconds)
            row["attempts"] = attempt
            if row["ok"]:
                break
            # A failed attempt is usually environmental, not the file: TH08 allows
            # one instance per session (named mutex) and needs a moment to release
            # it, so the *next* launch can lose the menu race.  Give the previous
            # process time to disappear before retrying.
            time.sleep(3.0 * attempt)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for row in rows if row["ok"])
    print(f"\n{ok}/{len(rows)} playable; written {out}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
