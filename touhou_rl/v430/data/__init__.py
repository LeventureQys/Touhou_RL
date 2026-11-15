"""v4.3.0 data package — W09 owns the clean hold-out lineage registry.

Layout (``Document/Update/Alpha/v4.3.0/DEVELOPMENT_GUIDE.md`` §2):

* :mod:`touhou_rl.v430.data.lineage` — the task-ancestor model, the
  *pretraining-source* register, the fail-closed ``validate_splits`` checker and
  the stable ``split_registry_sha`` that a dataset manifest binds as
  ``split_registry_sha`` (manifest field list, DEVELOPMENT_GUIDE §6.1).
  Standard library only: no numpy, no torch, no engine, no pixel I/O.

Work package ownership: W09 (``EXECUTION_PLAN.md`` §2).  Sibling modules
(``manifest``/``index``/stream loaders) belong to W03 and are **not** created
here; importing them fails honestly instead of returning a placeholder pass.

Nothing in this package launches the game, opens an asset, trains or evaluates.
"""

from __future__ import annotations

__all__ = [
    "PACKAGE",
]

#: Metadata only.  It never claims that a gate passed or that a module exists.
PACKAGE: dict[str, object] = {
    "package": "touhou_rl.v430.data",
    "work_package": "v4.3.0 W09",
    "modules": ["lineage"],
    "planned_modules_other_packages": ["manifest", "index", "stream"],
    "validator_entry": "touhou_rl.v430.data.lineage:validate_splits",
    "cli_subcommand": "validate-splits",
}
