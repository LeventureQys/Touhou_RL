"""Read-only auditor for the v8.30.0 training-host receipt.

Two checks in one pass:

  * consistency -- section numbering, cross references, superseded strings,
    key facts present, table integrity, control characters;
  * coverage    -- every sub-question of V830-ROUTE-INFO-01's six "minimum
    answer" cells has a corresponding answer in the receipt.

Both read the receipt as text and touch nothing.  Run from the repo root:

    python -B tools/rl/v8_30_receipt_audit.py
    python -B tools/rl/v8_30_receipt_audit.py --receipt <path> --json

Exit code 0 means both checks passed.  A passing audit means the document is
internally consistent and covers the questions asked; it is NOT experimental
validation of any claim inside it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_RECEIPT = Path(
    "Document/Update/Alpha/v8.30.0/TRAINING_HOST_REPLY_01.md")

#: Strings that must only survive inside the self-correction log.
SUPERSEDED = [
    "3 \u6b8b\u673a\u3001bomb \u6309\u673a\u4f53",
]

#: Key facts that must appear, with what each one pins down.
FACTS = {
    "13,195": "clear frames",
    "2,631,430": "clear score",
    "15,564": "negative-control frames",
    "19887": "max-frames",
    "52112": "bytes per decision",
    "ef2fbd15385f26f4": "obs-v3 digest prefix",
    "691bfd77b7eca336": "engine sha prefix",
    "48bf7621da5879ef": "live cfg sha prefix",
    "cfa292ae10ee1907": "mute-only sha prefix",
    "37b3e21d930adbdc": "audio audit blob prefix",
    "1523df26ec643b72": "th08-rl build commit",
    "381-389": "GameManager edge test",
}

#: The ticket's "minimum answer" cells, expanded to (item, sub-question, probe).
COVERAGE: list[tuple[str, str, str]] = [
    ("1", "stage or explicit gap", "th08-stage1-normal-team0"),
    ("1", "start event", "RESET"),
    ("1", "completion event", "stage_clear"),
    ("1", "takeover / first controlled frame", "ResetFrameCount"),
    ("1", "ship", "Border Team"),
    ("1", "legal actions / tempo", "18"),
    ("1", "shot & bomb convention", "SHOOT"),
    ("1", "initial & inherited resources", "livesRemaining"),
    ("2", "one success witness", "13195"),
    ("2", "one failure witness", "160"),
    ("2", "run id / path", "clear_reachability.json"),
    ("2", "action constraint", "shoot_pulse_period"),
    ("2", "terminal / resources", "deaths"),
    ("2", "complete vs first-death vs fault split", "death_stop"),
    ("3", "legal prefix exists?", "frame 92"),
    ("3", "continuous pixels", "obs-v3"),
    ("3", "real executed action history", "ACTION_HISTORY_SHAPE"),
    ("3", "memory rebuild at takeover", "\u9636\u6bb5\u5207\u6362"),
    ("3", "first missing frame / mask conflict", "8,664"),
    ("3", "visibility known/unknown", "\u53ef\u89c1\u6027"),
    ("4", "which controller for solvability", "\u6559\u5e08\u5f55\u50cf\u5e26"),
    ("4", "pixel candidate / fixed actor", "\u65e0"),
    ("4", "privileged-bomb hybrid marked", "\u7279\u6743"),
    ("5", "how many roots", "360"),
    ("5", "exposure / overlap / unknown", "used_for_training"),
    ("5", "reserved uses", "confirm"),
    ("6", "controlled frames", "19887"),
    ("6", "prefix/recovery/render/total wall clock", "115"),
    ("6", "record identity", "sha256"),
    ("6", "available window", "\u672a\u77e5"),
]

CATEGORY_WORDS = {
    "verified_this_pass": "\u672c\u6b21\u53ea\u8bfb\u6838\u5b9e",
    "historical": "\u5386\u53f2\u8bb0\u5f55",
    "gap": "\u7f3a\u53e3",
    "research_end": "\u7814\u7a76\u7aef",
}


def audit(text: str) -> dict:
    lines = text.splitlines()
    problems: list[str] = []

    nums = [int(m.group(1)) for m in
            (re.match(r"^## (\d+)\.", l) for l in lines) if m]
    if nums != list(range(len(nums))):
        problems.append(f"sections not contiguous: {nums}")

    refs = sorted({int(x) for x in re.findall(r"\u00a7(\d+)", text)})
    dangling = [r for r in refs if r not in nums]
    if dangling:
        problems.append(f"dangling section refs: {dangling}")

    superseded = {}
    for s in SUPERSEDED:
        hits = [i + 1 for i, l in enumerate(lines) if s in l]
        quoted = [i for i in hits if "\u66f4\u6b63" in lines[i - 1]]
        superseded[s] = {"lines": hits, "quoted_in_correction": quoted}
        if len(quoted) != len(hits):
            problems.append(f"superseded string outside correction log: {s!r} at {hits}")

    missing_facts = [k for k in FACTS if k not in text]
    if missing_facts:
        problems.append(f"missing facts: {missing_facts}")

    malformed = [i + 1 for i, l in enumerate(lines)
                 if l.startswith("|") and l.count("|") < 2]
    if malformed:
        problems.append(f"malformed table rows: {malformed}")

    ctrl = [hex(ord(c)) for c in text if ord(c) < 32 and c not in "\n\t"]
    if ctrl:
        problems.append(f"control characters: {ctrl[:5]}")

    uncovered = [(i, q, p) for i, q, p in COVERAGE if p not in text]
    if uncovered:
        problems.append(f"uncovered sub-questions: {[(i, q) for i, q, _ in uncovered]}")

    return {
        "sections": nums,
        "cross_refs": refs,
        "dangling_refs": dangling,
        "superseded": superseded,
        "missing_facts": missing_facts,
        "malformed_rows": malformed,
        "control_chars": len(ctrl),
        "coverage": {
            "total": len(COVERAGE),
            "covered": len(COVERAGE) - len(uncovered),
            "per_item": {
                item: sum(1 for i, _q, p in COVERAGE
                          if i == item and p in text)
                for item in sorted({c[0] for c in COVERAGE})
            },
        },
        "categories": {k: text.count(v) for k, v in CATEGORY_WORDS.items()},
        "problems": problems,
        "ok": not problems,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    path = Path(args.receipt)
    if not path.is_file():
        print(f"receipt not found: {path}", file=sys.stderr)
        return 2

    report = audit(path.read_text(encoding="utf-8"))
    report["receipt"] = str(path)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        print(f"receipt: {path}")
        print(f"sections: {report['sections']}")
        print(f"cross-refs: {report['cross_refs']} "
              f"dangling: {report['dangling_refs']}")
        for s, d in report["superseded"].items():
            print(f"superseded {s!r}: lines={d['lines']} "
                  f"quoted_in_correction={d['quoted_in_correction']}")
        print(f"missing facts: {report['missing_facts']}")
        print(f"malformed rows: {report['malformed_rows']}  "
              f"control chars: {report['control_chars']}")
        print(f"coverage: {report['coverage']['covered']}/"
              f"{report['coverage']['total']} {report['coverage']['per_item']}")
        print(f"categories: {report['categories']}")
        if report["problems"]:
            print("\nPROBLEMS:")
            for p in report["problems"]:
                print("  -", p)
        else:
            print("\nAUDIT OK: consistent and covering every sub-question")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
