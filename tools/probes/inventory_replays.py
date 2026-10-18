"""Inventory a directory of TH08 replays: metadata, duplicates, coverage gaps.

Why this exists
---------------
"Can I use these replays?" has three separate answers and they are easy to
conflate:

1. **Is the file intact?** - checksum, extents, LZSS payload, retail executable
   identity.  That is what ``third_party/th08/scripts/replay_file.py`` validates
   (it mirrors the reader this repo reconstructs), and
   ``tools/probes/check_replays.py`` confirms the runtime side by actually
   playing the file back.
2. **Is it already in the corpus?** - a byte-identical file adds nothing; a
   *different recording of the same cell* still adds a trajectory.
3. **What does it add?** - difficulty / shot / route (Final A vs B), input
   record counts (which bound how much wall-clock capture it costs) and whether
   the (shot, difficulty, final) cell is new or duplicated.

Usage::

    python tools/probes/inventory_replays.py --dir th08_replay
    python tools/probes/inventory_replays.py --dir th08_replay --out inventory.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPLAY_SCRIPTS = ROOT / "third_party/th08/scripts"
if str(REPLAY_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REPLAY_SCRIPTS))

from replay_file import DIFFICULTIES, SHOTS, inspect  # noqa: E402

MANIFEST = ROOT / "third_party/th08/config/replay-fixtures.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory TH08 replay files")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    fixtures = json.loads(manifest_path.read_text(encoding="utf-8"))["fixtures"] if manifest_path.exists() else []
    known_hashes = {f["sha256"] for f in fixtures}
    known_cells = {(f["shot"], f["difficulty"], f["stages"][-1]["index"]) for f in fixtures}

    rows = []
    for path in sorted(Path(args.dir).glob("*.rpy")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row = {"file": path.name, "bytes": path.stat().st_size, "sha256": digest}
        try:
            info = inspect(path)
        except Exception as exc:
            row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
            continue
        shot = SHOTS.index(info["shotName"])
        difficulty = DIFFICULTIES.index(info["difficultyName"])
        final = info["stages"][-1]["index"]
        frames = sum(stage["inputRecords"] for stage in info["stages"])
        row.update(
            {
                "ok": True,
                "shot": shot,
                "shotName": info["shotName"],
                "difficulty": difficulty,
                "difficultyName": info["difficultyName"],
                "final": "B" if final == 7 else ("Extra" if final == 8 else "A"),
                "practice": info["practice"],
                "clearState": info["clearState"],
                "stages": [stage["index"] for stage in info["stages"]],
                "frames": frames,
                "duplicate_of_fixture": digest in known_hashes,
                "cell_new": (shot, difficulty, final) not in known_cells,
            }
        )
        rows.append(row)

    ok = [row for row in rows if row.get("ok")]
    # Duplicates *inside* the directory matter too: two byte-identical files are
    # one trajectory, and keeping both only doubles capture cost for no gain.
    seen: dict[str, str] = {}
    for row in ok:
        first = seen.setdefault(row["sha256"], row["file"])
        row["duplicate_of_local"] = None if first == row["file"] else first
    print(f"| file | shot | difficulty | final | stages | frames | cleared | intact | dup | cell |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        if not row.get("ok"):
            print(f"| {row['file']} | - | - | - | - | - | - | **FAIL: {row['error']}** | - | - |")
            continue
        duplicate = row["duplicate_of_local"] or (
            "fixture" if row["duplicate_of_fixture"] else ""
        )
        print(
            f"| {row['file']} | {row['shotName']} | {row['difficultyName']} | {row['final']} | "
            f"{len(row['stages'])} | {row['frames']} | {'yes' if row['clearState'] else 'no'} | yes | "
            f"{duplicate or '-'} | "
            f"{'new' if row['cell_new'] else 'exists'} |"
        )

    print()
    for difficulty in range(len(DIFFICULTIES)):
        group = [row for row in ok if row["difficulty"] == difficulty]
        if not group:
            continue
        frames = sum(row["frames"] for row in group)
        # A replay records one input word per 60 Hz logic frame, so capture costs
        # frames / 60 seconds of wall clock, i.e. frames / 216000 hours.
        print(
            f"{DIFFICULTIES[difficulty]}: {len(group)} file(s), {frames:,} frames "
            f"= {frames / 216_000:.2f} h of capture at 60 fps"
        )
    lunatic_frames = sum(row["frames"] for row in ok if row["difficultyName"] == "Lunatic")
    if lunatic_frames:
        print(f"Lunatic adds {lunatic_frames / 2_979_674 * 100:.1f}% over the 24-replay corpus (2,979,674 frames)")
    print(f"intact: {len(ok)}/{len(rows)}; byte-identical to an existing fixture: "
          f"{sum(1 for row in ok if row['duplicate_of_fixture'])}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {args.out}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
