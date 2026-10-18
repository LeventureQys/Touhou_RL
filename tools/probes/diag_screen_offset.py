"""Locate TitleScreen::currentScreen empirically by diffing the struct."""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from touhou_rl import menu, spec  # noqa: E402
from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir  # noqa: E402


def dump_words(session: GameSession, start: int, length: int) -> list[int]:
    mem = session.mem
    assert mem is not None
    title = mem.u32(mem.module_base - spec.IMAGE_BASE + spec.ADDR_TITLE_SCREEN)
    raw = mem.read(title + start, length)
    return list(struct.unpack(f"<{length // 4}I", raw))


def scan_range(session: GameSession) -> dict:
    mem = session.mem
    assert mem is not None
    title = mem.u32(mem.module_base - spec.IMAGE_BASE + spec.ADDR_TITLE_SCREEN)
    return {"title_ptr": hex(title), "c200": dump_words(session, 0xC200, 0x1400), "tail": dump_words(session, 0x14000, 0x500)}


def main() -> int:
    out: dict = {}
    base = ROOT / "third_party/th08/build/play"
    run = prepare_run_dir(ROOT / "third_party/th08/build/rl/diag-screen", base, windowed=True)
    session = GameSession(SessionOptions(run_dir=run))
    try:
        session.start()
        time.sleep(2.0)
        before = scan_range(session)
        st = session.read_state(with_bullets=False)
        out["before"] = {"screen_state": st.title_screen_state, "cursor": st.title_cursor}
        menu.tap(session, "z", settle_s=1.2)
        st = session.read_state(with_bullets=False)
        out["after"] = {"screen_state": st.title_screen_state, "cursor": st.title_cursor}
        after = scan_range(session)
        # A field that flips 0 -> 4 is the screen enum (StartMenu -> DifficultySelect).
        candidates = []
        for region in ("c200", "tail"):
            for i, (a, b) in enumerate(zip(before[region], after[region])):
                if a != b and b in (0, 4, 5, 7, 8, 9, 11, 12, 13):
                    candidates.append({"region": region, "offset": hex(0xC200 + i * 4) if region == "c200" else hex(0x14000 + i * 4), "before": a, "after": b})
        out["candidates"] = candidates
        out["changed_count"] = {
            region: sum(1 for a, b in zip(before[region], after[region]) if a != b) for region in ("c200", "tail")
        }
        # Also record where cursor lives (should not change on this transition).
        out["title_ptr_same"] = before["title_ptr"] == after["title_ptr"]
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        out["traceback"] = traceback.format_exc()
    finally:
        session.close()
    (ROOT / "third_party/th08/build/rl/diag-screen.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
