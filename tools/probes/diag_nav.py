"""Diagnose the SendInput menu flow and background replay playback."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from touhou_rl import menu, spec  # noqa: E402
from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir  # noqa: E402

REPLAY = ROOT / "third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/th8_ud2945.rpy"


def snapshot(session: GameSession) -> dict:
    st = session.read_state(with_bullets=False)
    return {
        "sup": st.supervisor_state,
        "screen": st.title_screen_current,
        "screen_state": st.title_screen_state,
        "cursor": st.title_cursor,
        "stage": st.stage,
        "frame": st.gameplay_frame,
        "lives": st.lives,
        "bombs": st.bombs,
        "is_replay": st.is_replay,
        "is_demo": st.is_demo,
        "replay_mode": st.replay_mode,
    }


def main() -> int:
    out: dict = {"steps": []}
    base = ROOT / "third_party/th08/build/play"
    run = prepare_run_dir(ROOT / "third_party/th08/build/rl/diag-nav", base, windowed=True)
    menu.install_replay(run, REPLAY)
    session = GameSession(SessionOptions(run_dir=run))
    try:
        session.start()
        out["client_size"] = session.client_size
        time.sleep(1.5)
        out["steps"].append({"t": "boot", **snapshot(session)})

        # 1) navigate into a Lunatic game via SendInput, measuring each screen.
        menu.ensure_title(session)
        out["steps"].append({"t": "ensure_title", **snapshot(session)})
        menu.tap(session, "down", settle_s=0.2)
        out["steps"].append({"t": "cursor_down", **snapshot(session)})
        menu.tap(session, "up", settle_s=0.2)
        out["steps"].append({"t": "cursor_up", **snapshot(session)})
        menu.enter_game(session, difficulty=spec.LUNATIC, team=0)
        time.sleep(1.5)
        out["steps"].append({"t": "in_game", **snapshot(session)})
        first = session.read_state(with_bullets=False).gameplay_frame
        time.sleep(4.0)
        second = session.read_state(with_bullets=False).gameplay_frame
        out["logic_fps_foreground"] = round((second - first) / 4.0, 2)

        # 2) check that the game keeps running when the window is not foreground.
        import ctypes
        from ctypes import wintypes as wt

        from touhou_rl import winapi

        shell = ctypes.windll.shell32
        foreground_before = winapi.user32.GetForegroundWindow()
        shell.ShellExecuteW(None, "open", "notepad.exe", None, None, 1)
        time.sleep(1.2)
        out["foreground_changed"] = winapi.user32.GetForegroundWindow() != foreground_before
        f1 = session.read_state(with_bullets=False).gameplay_frame
        time.sleep(3.0)
        f2 = session.read_state(with_bullets=False).gameplay_frame
        out["logic_fps_background"] = round((f2 - f1) / 3.0, 2)
        # close notepad we opened
        import subprocess

        subprocess.run(["taskkill", "/IM", "notepad.exe", "/F"], capture_output=True)

        # 3) replay playback flow: back to title, then Replay menu.
        menu.exit_run_to_title(session)
        out["steps"].append({"t": "exit_to_title", **snapshot(session)})
        menu.enter_replay(session, stage=0)
        out["steps"].append({"t": "replay_started", **snapshot(session)})
        r1 = session.read_state(with_bullets=False).gameplay_frame
        time.sleep(3.0)
        r2 = session.read_state(with_bullets=False)
        out["replay"] = {
            "frames_in_3s": (r2.gameplay_frame - r1) & 0xFFFFFFFF,
            "is_replay": r2.is_replay,
            "stage": r2.stage,
            "bullets": r2.active_bullets,
            "score": r2.score,
        }
        out["input_mask_during_replay"] = r2.input_mask
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        out["traceback"] = traceback.format_exc()
        out["steps"].append({"t": "error", **snapshot(session)})
    finally:
        session.close()
    (ROOT / "third_party/th08/build/rl/diag-nav.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2)[:8000])
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
