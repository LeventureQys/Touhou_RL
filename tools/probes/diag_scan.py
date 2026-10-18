"""Find the live GameManager counter offsets by scanning for frame increments."""

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

WINDOW_START = 0x3D000
WINDOW_SIZE = 0x1400


def dump(session: GameSession) -> list[int]:
    mem = session.mem
    assert mem is not None
    raw = mem.read(mem.module_base - spec.IMAGE_BASE + spec.ADDR_GAME_MANAGER + WINDOW_START, WINDOW_SIZE)
    return list(struct.unpack(f"<{WINDOW_SIZE // 4}I", raw))


def main() -> int:
    out: dict = {}
    base = ROOT / "third_party/th08/build/play"
    run = prepare_run_dir(ROOT / "third_party/th08/build/rl/diag-scan", base, windowed=True)
    session = GameSession(SessionOptions(run_dir=run))
    try:
        session.start()
        time.sleep(1.0)
        menu.enter_game(session, difficulty=spec.LUNATIC, team=0)
        time.sleep(2.0)
        a = dump(session)
        time.sleep(0.5)
        b = dump(session)
        incs = []
        for i, (x, y) in enumerate(zip(a, b)):
            if y > x and 5 <= (y - x) <= 60:
                incs.append({"offset": hex(WINDOW_START + i * 4), "a": x, "b": y, "delta": y - x})
        out["incrementing"] = incs[:60]
        # neighbours around the biggest plausible counter (>=20 frames in 0.5 s)
        out["near"] = {}
        for row in incs:
            if row["delta"] >= 20:
                off = int(row["offset"], 16)
                idx = (off - WINDOW_START) // 4
                out["near"][row["offset"]] = {
                    f"{(WINDOW_START + (idx + k) * 4):#x}": dumps
                    for k, dumps in enumerate(range(-4, 12))
                }
                break
        out["flags_region"] = [hex(v) for v in dump(session)[(0x3DBAC - WINDOW_START) // 4: (0x3DBAC - WINDOW_START) // 4 + 24]]
        # Direct scalar reads via the same API for cross-checking.
        mem = session.mem
        assert mem is not None
        gm = mem.module_base - spec.IMAGE_BASE + spec.ADDR_GAME_MANAGER
        out["direct"] = {
            "gm_ptr0x8": hex(mem.u32(gm + 0x8)),
            "flags_0x3DBAC": hex(mem.u32(gm + 0x3DBAC)),
            "stage_0x3DDC4": mem.i32(gm + 0x3DDC4),
            "frame_0x3DDC0": mem.u32(gm + 0x3DDC0),
        }
        st = session.read_state(with_bullets=False)
        out["reader"] = {
            "flags": hex(st.flags),
            "stage": st.stage,
            "frame": st.gameplay_frame,
            "lives": st.lives,
            "player": [st.player_x, st.player_y],
        }
        # Real player position sanity: should be inside the playfield.
        out["state_probe"] = {"focus": st.focus_mode, "active_bullets": st.active_bullets}
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        out["traceback"] = traceback.format_exc()
    finally:
        session.close()
    (ROOT / "third_party/th08/build/rl/diag-scan.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "near"}, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
