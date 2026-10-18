"""Compare direct vs block GameManager reads and locate the live frame counter."""

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


def main() -> int:
    out: dict = {}
    base = ROOT / "third_party/th08/build/play"
    run = prepare_run_dir(ROOT / "third_party/th08/build/rl/diag-game", base, windowed=True)
    session = GameSession(SessionOptions(run_dir=run))
    try:
        session.start()
        time.sleep(1.5)
        menu.ensure_title(session)
        for _ in range(3):
            menu.tap(session, "z", settle_s=0.6)
        time.sleep(2.5)
        mem = session.mem
        assert mem is not None
        gm = mem.module_base - spec.IMAGE_BASE + spec.ADDR_GAME_MANAGER
        out["gm_base"] = hex(gm)
        start = 0x3C000
        size = 0x2400
        a = struct.unpack(f"<{size // 4}I", mem.read(gm + start, size))
        time.sleep(0.5)
        b = struct.unpack(f"<{size // 4}I", mem.read(gm + start, size))
        incs = [
            {"offset": hex(start + i * 4), "a": x, "b": y, "delta": y - x}
            for i, (x, y) in enumerate(zip(a, b))
            if y > x and 5 <= (y - x) <= 60
        ]
        out["incrementing"] = incs[:40]
        out["direct"] = {
            "flags_0x3DBAC": hex(mem.u32(gm + 0x3DBAC)),
            "stage_0x3DDC4": mem.i32(gm + 0x3DDC4),
            "frame_0x3DDC0": mem.u32(gm + 0x3DDC0),
            "globals_ptr": hex(mem.u32(gm + 0x8)),
        }
        st = session.read_state(with_bullets=False)
        out["reader"] = {
            "flags": hex(st.flags),
            "stage": st.stage,
            "frame": st.gameplay_frame,
            "lives": st.lives,
            "bombs": st.bombs,
            "bullets": st.active_bullets,
            "player": [round(st.player_x, 2), round(st.player_y, 2)],
            "focus": st.focus_mode,
            "input_mask": hex(st.input_mask),
        }
        # Dump a window around the claimed flags offset for manual inspection.
        words = struct.unpack("<64I", mem.read(gm + 0x3DB90, 256))
        out["words_around_flags"] = {hex(0x3DB90 + i * 4): hex(v) for i, v in enumerate(words)}
        # Bullet sanity via the bullet reader.
        field = session.reader.read_bullets() if session.reader else None
        if field is not None:
            mask = field.active_mask()
            pos = field.position[mask]
            out["bullets"] = {
                "active": int(mask.sum()),
                "x_range": [round(float(pos[:, 0].min()), 1), round(float(pos[:, 0].max()), 1)] if len(pos) else None,
                "y_range": [round(float(pos[:, 1].min()), 1), round(float(pos[:, 1].max()), 1)] if len(pos) else None,
            }
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        out["traceback"] = traceback.format_exc()
    finally:
        session.close()
    (ROOT / "third_party/th08/build/rl/diag-game.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
