"""Minimal reader comparison: block buffer bytes vs GameState fields."""

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
from touhou_rl import state as state_mod  # noqa: E402


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
        reader = session.reader
        assert reader is not None
        st = reader.read_state(with_bullets=False)
        out["state_flags"] = hex(st.flags)
        out["state_frame"] = st.gameplay_frame
        out["state_stage"] = st.stage
        out["block_start_const"] = hex(state_mod._GM_BLOCK_START)
        out["block_size_const"] = state_mod._GM_BLOCK_SIZE
        out["gm_attr"] = hex(reader._gm)
        # Re-read through the very same buffer right after read_state.
        reader.mem.read_into(reader._gm, reader._gm_buf)
        raw = reader._gm_buf.raw
        out["raw_len"] = len(raw)
        out["raw_head"] = raw[:16].hex()
        out["raw_index4"] = hex(struct.unpack_from("<I", raw, 4)[0])
        out["raw_frame_index"] = struct.unpack_from(
            "<I", raw, state_mod.spec.OFF_GM_GAMEPLAY_FRAME - state_mod._GM_BLOCK_START
        )[0]
        out["reader_reads"] = reader.mem.reads
        out["reader_failures"] = reader.mem.failures
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        out["traceback"] = traceback.format_exc()
    finally:
        session.close()
    (ROOT / "third_party/th08/build/rl/diag-reader2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
