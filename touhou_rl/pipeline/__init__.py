"""Three-stage data pipeline for Touhou_RL.

``collect``  game -> raw shards (the only stage that costs game wall-clock)
``process``  validate + teacher labels + split pinning -> ``manifest.json``
``train``    manifest -> checkpoint + ``train-report.json``

See ``README.md`` in this directory for the stage guarantees and the command line.
"""

from __future__ import annotations

__all__ = ["manifest", "collect", "process", "train", "cli"]
