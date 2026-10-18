"""Entry point so the pipeline can be driven as ``python -m touhou_rl.pipeline``.

Kept separate from :mod:`touhou_rl.pipeline.cli` because a package's ``__main__``
must be a module, not the package itself; the CLI module stays importable so
tests can call ``build_parser()`` without spawning a process.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
