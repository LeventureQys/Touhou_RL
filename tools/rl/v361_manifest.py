"""Evidence manifest for v3.6.1 (same dumb bookkeeping as v3.6.0)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/rl/v361_manifest.py")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.root
    files = []
    excluded = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.parent.name in ("trace", "bullets") and "probe" in path.parts:
            excluded.append(str(path.relative_to(root)).replace("\\", "/"))
            continue
        files.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema": "v361-evidence-manifest/1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(root).replace("\\", "/"),
        "files": files,
        "excluded_caches": excluded,
        "totals": {"files": len(files), "bytes": sum(item["bytes"] for item in files)},
    }
    target = root / "evidence" / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(target), "files": len(files)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
