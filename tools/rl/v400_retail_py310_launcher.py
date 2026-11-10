"""Launch the unchanged upstream retail capture under Python 3.10.

Only supply hashlib.file_digest's streaming SHA-256 compatibility API; never
patch the retail binary, replay, upstream script, or game data. Python >=3.11
uses the standard implementation unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import runpy
import sys
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parents[2] / "third_party/th08/scripts/capture-replay.py"


def _file_digest(stream, digest, /, *, _bufsize=262144):
    h = hashlib.new(digest) if isinstance(digest, str) else digest()
    while data := stream.read(_bufsize):
        h.update(data)
    return h


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--shim-self-test" in args:
        p = argparse.ArgumentParser()
        p.add_argument("--shim-self-test", action="store_true")
        p.parse_args(args)
        from io import BytesIO
        assert _file_digest(BytesIO(b"test"), "sha256").hexdigest() == hashlib.sha256(b"test").hexdigest()
        print("hashlib.file_digest compatibility shim verified")
        return 0
    if not hasattr(hashlib, "file_digest"):
        hashlib.file_digest = _file_digest
    # runpy preserves upstream file and directory-relative imports.
    sys.path.insert(0, str(UPSTREAM.parent))
    sys.argv = [str(UPSTREAM), *args]
    runpy.run_path(str(UPSTREAM), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
