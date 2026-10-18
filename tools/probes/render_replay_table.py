"""Render the replay corpus as a Markdown section for the root ``readme.md``.

Why this exists
---------------
The replay inventory is machine-checked (``inventory_replays.py`` writes
``replay-inventory.json``); the readme is hand written.  Copying 40 rows from one
into the other by hand invites silent drift, so the table is generated from the
validated JSON instead.  Re-run this whenever the corpus changes::

    python tools/probes/inventory_replays.py --dir th08_replay \
        --out third_party/th08/build/rl/probe-v103/replay-inventory.json
    python tools/probes/render_replay_table.py \
        --inventory third_party/th08/build/rl/probe-v103/replay-inventory.json \
        > replay-section.md

Reported per file: shot, difficulty, route (Final A / Final B / Extra) and the
input record count (one record per 60 Hz logic frame; 3600 records = 1 minute of
play).

Deliberately *not* reported as a "cleared" column: ``clearState`` is 2 for every
file here, but ``ReplayManager.cpp:453`` forces it to 2 whenever ``shotType > 3``
(the eight solo characters), so the field only proves a no-continue Final B clear
for the four team shots.  A column of identical 2s would invite the reader to
conclude "all 40 are 1CC" for replays where the game does not record that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Lunatic first: it is the difficulty this project is actually chasing.
DIFFICULTY_ORDER = ["Lunatic", "Hard", "Normal", "Easy", "Extra"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render replay inventory as Markdown")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--out",
        default="",
        help="write UTF-8 here instead of stdout (the Windows console mangles CJK output)",
    )
    args = parser.parse_args(argv)

    rows = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("ok")]
    rows.sort(
        key=lambda row: (
            DIFFICULTY_ORDER.index(row["difficultyName"])
            if row["difficultyName"] in DIFFICULTY_ORDER
            else 99,
            row["shot"],
            row["file"].lower(),
        )
    )

    lines: list[str] = []
    if args.title:
        lines += [f"### {args.title}", ""]
    lines += ["| 文件 | 机体 | 难度 | 路线 | 输入帧数 | 时长 |", "| --- | --- | --- | --- | ---: | ---: |"]
    totals: dict[str, list[int]] = {}
    for row in rows:
        frames = int(row["frames"])
        totals.setdefault(row["difficultyName"], []).append(frames)
        lines.append(
            f"| `{row['file']}` | {row['shotName']} | {row['difficultyName']} | {row['final']} | "
            f"{frames:,} | {frames / 3600:.1f} 分钟 |"
        )

    lines += ["", "| 难度 | 份数 | 总输入帧 | 合计时长 |", "| --- | ---: | ---: | ---: |"]
    grand_files = grand_frames = 0
    for difficulty in DIFFICULTY_ORDER:
        group = totals.get(difficulty)
        if not group:
            continue
        summary = sum(group)
        grand_files += len(group)
        grand_frames += summary
        lines.append(f"| {difficulty} | {len(group)} | {summary:,} | {summary / 3600:.1f} 分钟 |")
    lines.append(
        f"| **合计** | **{grand_files}** | **{grand_frames:,}** | "
        f"**{grand_frames / 3600:.1f} 分钟（{grand_frames / 216_000:.1f} 小时）** |"
    )

    text = "\n".join(lines) + "\n"
    if args.out:
        # Explicit UTF-8 with LF newlines: the Windows console codepage mangles CJK
        # output, and Readme already uses LF.
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        print(f"written: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
