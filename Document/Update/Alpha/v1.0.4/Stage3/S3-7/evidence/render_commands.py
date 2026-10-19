"""Render the formal-command table (section 12) from the dry-run JSON plans."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
EVIDENCE = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-7" / "evidence"


def main() -> int:
    lines: list[str] = []
    for phase in ("v3-matrix", "r2", "r3", "stability"):
        payload = json.loads((EVIDENCE / f"dryrun-{phase}.json").read_text(encoding="utf-8"))
        for plan in payload["plans"]:
            lines.append(f"[{phase}] {plan['experiment']} - {plan['estimated_wall_clock']}")
            lines.append(" ".join(plan["command"]))
            lines.append("")
    text = "\n".join(lines)
    (EVIDENCE / "dryrun-commands.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
