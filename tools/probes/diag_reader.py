"""Straight comparison of GameStateReader block reads vs direct scalar reads."""

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
from touhou_rl.state import _GM_BLOCK_START  # noqa: E402


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
        reader = session.reader
        assert mem is not None and reader is not None
        gm = mem.module_base - spec.IMAGE_BASE + spec.ADDR_GAME_MANAGER
        out["module_base"] = hex(mem.module_base)
        out["reader_gm"] = hex(reader._gm)
        out["direct_gm"] = hex(gm)
        out["block_start"] = hex(_GM_BLOCK_START)
        block = mem.read(gm + _GM_BLOCK_START, 32)
        out["direct_block_hex"] = block.hex()
        out["direct_flags"] = hex(struct.unpack_from("<I", block, 4)[0])
        out["direct_frame"] = struct.unpack_from("<I", block, 0x3DDC0 - _GM_BLOCK_START)[0]
        st = reader.read_state(with_bullets=False)
        out["reader_flags"] = hex(st.flags)
        out["reader_frame"] = st.gameplay_frame
        # Same read through the reader's own buffer object.
        reader.mem.read_into(reader._gm, reader._gm_buf)
        raw = reader._gm_buf.raw
        out["buffer_len"] = len(raw)
        out["buffer_hex_head"] = raw[:32].hex()
        out["buffer_flags"] = hex(struct.unpack_from("<I", raw, 4)[0])
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        out["traceback"] = traceback.format_exc()
    finally:
        session.close()
    (ROOT / "third_party/th08/build/rl/diag-reader.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
