"""v4.3.1 audit: are the deployed game data dirs actually muted?

The harness template written by `tools/rl/v400_make_cfg.py` sets
``playSounds = 0``, ``musicVolume = 0``, ``sfxVolume = 0``. But each capture run gets
its own *private* data directory, and the ``th08.cfg`` that lands in it is whatever
the engine wrote on exit -- which can differ from the template. This walks the data
directories on disk and decodes the two volume bytes plus ``playSounds`` so the
"game sound is off" claim can be checked instead of assumed.

Field offsets are from `tools/rl/v400_make_cfg.py` (Supervisor.hpp layout):
    0x1F musicMode   0x20 playSounds   0x27 musicVolume(i8)   0x28 sfxVolume(i8)

Read-only. Writes one JSON under Document/Update/Alpha/v4.3.1/evidence/R14/.
"""

from __future__ import annotations

import json
import pathlib
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R14"
ROOTS = [REPO / ".tmp_v420_data", REPO / ".tmp_v430_o2", REPO / ".tmp_v400_d1",
         REPO / ".tmp_v410_s1c", REPO / ".tmp_v410_wp05_round1"]
SIZE = 0x3C


def decode(path: pathlib.Path) -> dict | None:
    raw = path.read_bytes()
    if len(raw) != SIZE:
        return {"path": str(path), "error": f"size {len(raw)} != {SIZE}"}
    music_mode = raw[0x1F]
    play_sounds = raw[0x20]
    music_volume = struct.unpack_from("<b", raw, 0x27)[0]
    sfx_volume = struct.unpack_from("<b", raw, 0x28)[0]
    return {
        "path": str(path),
        "musicMode": music_mode,
        "playSounds": play_sounds,
        "musicVolume": music_volume,
        "sfxVolume": sfx_volume,
        "muted": (play_sounds == 0 and music_volume <= 0 and sfx_volume <= 0),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    found = []
    for root in ROOTS:
        if not root.is_dir():
            continue
        # only two levels deep: <root>/<case>/th08.cfg and <root>/.../data/th08.cfg
        for pattern in ("*/th08.cfg", "*/*/th08.cfg", "*/*/*/th08.cfg"):
            for p in root.glob(pattern):
                try:
                    d = decode(p)
                except OSError as exc:
                    d = {"path": str(p), "error": str(exc)}
                if d:
                    d["root"] = root.name
                    found.append(d)

    # de-duplicate by resolved path
    seen, uniq = set(), []
    for d in found:
        key = d["path"].lower()
        if key not in seen:
            seen.add(key)
            uniq.append(d)

    ok = [d for d in uniq if "error" not in d]
    muted = [d for d in ok if d["muted"]]
    loud = [d for d in ok if not d["muted"]]
    out = {
        "schema": "v431-audit-r14/1",
        "template_reference": {
            "source": "tools/rl/v400_make_cfg.py --dry-run",
            "playSounds": 0, "musicVolume": 0, "sfxVolume": 0, "musicMode": 0,
        },
        "counts": {"cfg_files": len(uniq), "decoded": len(ok),
                   "errors": len(uniq) - len(ok),
                   "muted": len(muted), "audible": len(loud)},
        "audible_examples": loud[:20],
        "muted_examples": muted[:5],
        "by_root": {},
    }
    for d in ok:
        b = out["by_root"].setdefault(d["root"], {"muted": 0, "audible": 0})
        b["muted" if d["muted"] else "audible"] += 1

    (OUT / "audio_cfg_audit.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"th08.cfg found: {len(uniq)}   decoded: {len(ok)}   errors: {len(uniq)-len(ok)}")
    print(f"  muted   (playSounds=0, volumes<=0): {len(muted)}")
    print(f"  audible (any of those not off)    : {len(loud)}")
    print()
    print(f"{'root':24s} {'muted':>6s} {'audible':>8s}")
    for root, c in out["by_root"].items():
        print(f"{root:24s} {c['muted']:6d} {c['audible']:8d}")
    print()
    for d in loud[:8]:
        print(f"  audible: playSounds={d['playSounds']} music={d['musicVolume']} "
              f"sfx={d['sfxVolume']}  {d['path'].split('Touhou_RL')[-1][:90]}")
    print()
    print(f"wrote: {OUT / 'audio_cfg_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
