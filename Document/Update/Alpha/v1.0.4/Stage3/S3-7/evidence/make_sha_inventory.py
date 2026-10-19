"""Write the S3-7 sha256 inventory (code, recipes, evidence, report).

Usage::

    python Document/Update/Alpha/v1.0.4/Stage3/S3-7/evidence/make_sha_inventory.py

Only files that exist are listed; the report's own digest is computed last so the
file does not have to describe itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
S37 = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-7"

TARGETS = [
    "touhou_rl/vision_train.py",
    "touhou_rl/vmodels2.py",
    "touhou_rl/tests/test_v104_refine.py",
    "tools/rl/refine_v104.py",
]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def rows() -> list[tuple[str, int, str]]:
    entries: list[tuple[str, int, str]] = []
    for relative in TARGETS:
        path = ROOT / relative
        entries.append((relative, path.stat().st_size, digest(path)))
    for path in sorted(S37.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "sha256sums.txt":
            continue
        entries.append((path.relative_to(ROOT).as_posix(), path.stat().st_size, digest(path)))
    return entries


def main() -> int:
    entries = rows()
    width = max(len(name) for name, _size, _sha in entries)
    lines = [f"{sha}  {size:>10}  {name.ljust(width)}" for name, size, sha in entries]
    target = S37 / "evidence" / "sha256sums.txt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(entries)} files -> {target.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
