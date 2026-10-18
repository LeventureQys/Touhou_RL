"""Step-by-step menu trace with raw flags, to validate entering a Lunatic run."""

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


def probe(session: GameSession) -> dict:
    st = session.read_state(with_bullets=False)
    mem = session.mem
    assert mem is not None
    delta = mem.module_base - spec.IMAGE_BASE
    gm = spec.ADDR_GAME_MANAGER + delta
    row = {
        "sup": st.supervisor_state,
        "screen": st.title_screen_current if st.supervisor_state == spec.STATE_TITLE_SCREEN else None,
        "screen_state": st.title_screen_state if st.supervisor_state == spec.STATE_TITLE_SCREEN else None,
        "cursor": st.title_cursor if st.supervisor_state == spec.STATE_TITLE_SCREEN else None,
        "flags": hex(st.flags),
        "demo": st.is_demo,
        "replay": st.is_replay,
        "practice": st.is_practice,
        "stage": st.stage,
        "frame": st.gameplay_frame,
        "lives": round(st.lives, 3),
        "bombs": round(st.bombs, 3),
        "power": round(st.power, 3),
        "replay_mode": st.replay_mode,
        "character": st.character,
        "shot": st.shot_type,
        "replay_filename": mem.read(gm + 0x3DBBC, 40).split(b"\x00")[0][:40].decode("ascii", "replace"),
    }
    return row


def main() -> int:
    out: dict = {"steps": []}
    base = ROOT / "third_party/th08/build/play"
    run = prepare_run_dir(ROOT / "third_party/th08/build/rl/diag-game", base, windowed=True)
    session = GameSession(SessionOptions(run_dir=run))
    try:
        session.start()
        time.sleep(1.5)
        out["steps"].append({"t": "boot", **probe(session)})
        menu.ensure_title(session)
        menu.tap(session, "z", settle_s=0.6)
        out["steps"].append({"t": "after_start_z", **probe(session)})
        menu.tap(session, "z", settle_s=0.6)
        out["steps"].append({"t": "after_difficulty_z", **probe(session)})
        menu.tap(session, "z", settle_s=0.6)
        for i in range(8):
            time.sleep(0.5)
            out["steps"].append({"t": f"after_character_z+{(i + 1) * 0.5:.1f}s", **probe(session)})
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        out["traceback"] = traceback.format_exc()
        try:
            out["steps"].append({"t": "error", **probe(session)})
        except Exception:
            pass
    finally:
        session.close()
    (ROOT / "third_party/th08/build/rl/diag-game.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for row in out["steps"]:
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
