"""v3.7.0 ``escape`` package: space-time route planning for wide bullet curtains.

Why this package exists
-----------------------
The v3.6.x line ended with seven tactical probes, of which none was allowed into
the control loop; the one that looked best on its own metric (``CollectPower``)
doubled the death rate on the live game.  The v3.7.0 need statement is narrower
and harder: *choose a reachable exit before the curtain closes*, instead of
dodging the nearest bullet harder.

This package is deliberately isolated from the old controller.  ``pilot.py``
stays the frozen baseline (a 20-frame, first-action beam search); nothing in
``escape`` imports it at decision time, and nothing here is wired into
``env.py`` yet.  The first version of the package only contains what D0/R1 must
freeze before any planner code is written:

* :mod:`touhou_rl.escape.contracts` -- units, records and the generated JSON
  schema; the schema file in ``Document/Update/Alpha/v3.7.0/contract/`` is
  produced from it, not hand-edited.
* :mod:`touhou_rl.escape.geometry` -- the player movement model measured on the
  real game and an analytic swept box-vs-box clearance, because the old pilot's
  per-step circle sampling can miss a crossing that happens between two samples.
* :mod:`touhou_rl.escape.baseline` -- the frozen old-pilot configuration and a
  digest-pinned golden fixture, so "the baseline did not change" is checkable
  without launching the game.
* :mod:`touhou_rl.escape.snapshot` -- code/worktree snapshot instead of a bare
  ``git HEAD``, because the tree had uncommitted modifications when D0 ran.

Nothing in this package reads memory, launches the game or trains a network.
"""

from __future__ import annotations

__all__ = [
    "CONTRACT_VERSION",
    "PACKAGE_STAGE",
]

# Bumped whenever a frozen record changes meaning; every emitted trace repeats it.
CONTRACT_VERSION = "v370-escape/1"

# Honest status flag: importers and reports can print it instead of claiming that
# a planner exists.  R2-D3 = contracts, baseline, adapter, testbed and forecast;
# no router/controller, no live game run.
PACKAGE_STAGE = "R2-D3: contract+baseline+adapter+testbed+forecast; no router, no game run"
