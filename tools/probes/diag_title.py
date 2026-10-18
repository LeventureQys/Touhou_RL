"""Diagnose TitleScreen offsets and the PostMessage input path."""

from __future__ import annotations

import ctypes
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from touhou_rl import spec, winapi  # noqa: E402
from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir  # noqa: E402

out: dict = {}


def dump(session: GameSession, label: str) -> dict:
    mem = session.mem
    assert mem is not None
    title = mem.u32(mem.module_base - spec.IMAGE_BASE + spec.ADDR_TITLE_SCREEN)
    st = session.read_state(with_bullets=False)
    row = {
        "label": label,
        "supervisor": st.supervisor_state,
        "title_ptr": hex(title),
        "cursor(0x0)": st.title_cursor,
        "cursor2(0x8)": st.title_cursor2,
        "screen_state(0xC)": st.title_screen_state,
        "practice(0x6C)": st.title_practice_state,
        "current(0x14448)": st.title_screen_current,
        "state(0x14454)": st.title_state,
        "replay_count": st.title_replay_count,
        "head_hex": mem.read(title, 0x18).hex() if title else "",
        "tail_hex": mem.read(title + 0x14440, 0x20).hex() if title else "",
        "replay_hex": mem.read(title + 0xC280, 0x20).hex() if title else "",
    }
    return row


def main() -> int:
    winapi.dpi_aware()
    base = ROOT / "third_party/th08/build/play"
    run = prepare_run_dir(ROOT / "third_party/th08/build/rl/diag-title", base, windowed=True)
    session = GameSession(SessionOptions(run_dir=run))
    rows = []
    try:
        session.start()
        rows.append(dump(session, "after start"))
        time.sleep(2.0)
        rows.append(dump(session, "after 2s"))
        # Post one Z and watch the reaction for 2 seconds.
        menu_taps = []
        for i in range(3):
            st = session.read_state(with_bullets=False)
            menu_taps.append({"i": i, "before": {"cur": st.title_cursor, "screen": st.title_screen_current,
                                                 "state": st.title_state, "sup": st.supervisor_state}})
            session.keyboard.tap("z", 0.08)
            time.sleep(0.8)
            st = session.read_state(with_bullets=False)
            menu_taps.append({"i": i, "after": {"cur": st.title_cursor, "screen": st.title_screen_current,
                                                "state": st.title_state, "sup": st.supervisor_state,
                                                "gameplay_frame": st.gameplay_frame}})
        rows.append(dump(session, "after 3 taps"))
        # Try direct WM_KEYDOWN without KEYUP to see if GetKeyboardState sees it.
        vk, scan = spec.KEYS["down"]
        winapi.user32.PostMessageW(session.hwnd, winapi.WM_KEYDOWN, vk, 1 | (scan << 16))
        time.sleep(0.3)
        st = session.read_state(with_bullets=False)
        down_state = {"cursor": st.title_cursor, "input_mask": st.input_mask}
        winapi.user32.PostMessageW(session.hwnd, winapi.WM_KEYUP, vk, 1 | (scan << 16) | 0xC0000000)
        out["direct_post"] = down_state
        rows.append(dump(session, "after direct post"))
        out["menu_taps"] = menu_taps
        out["rows"] = rows
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        out["traceback"] = traceback.format_exc()
        out["rows"] = rows
    finally:
        session.close()
    (ROOT / "third_party/th08/build/rl/diag-title.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
