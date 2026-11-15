"""v4.3.0 contract, action and (later) session/evaluation package.

This is the **root package** of the ``v430`` implementation.  It is owned by
work package W01 and deliberately contains *no* business logic: only the package
identity and the version constants a caller needs in order to name what it is
talking to.

Layout (``Document/Update/Alpha/v4.3.0/DEVELOPMENT_GUIDE.md`` §2):

* :mod:`touhou_rl.v430.contracts` — the frozen data contract: the 72-way joint
  action, the four independent action heads, the :class:`ActorObservation`
  shapes/dtypes, the O2 encoding parameters and digest, the two-frame time
  contract, the transition fields and the result-JSON/exit-code semantics.
  Standard library only (no numpy, no torch, no engine), fail-closed.
* :mod:`touhou_rl.v430.actions` — the 72-combination -> per-frame engine mask
  state machine, the bomb press-edge accounting and the single-owner key
  contract.  Sampling and evaluation share the one compiler.
* ``touhou_rl.v430.session`` / ``evaluate.py`` — **not implemented by this
  slice**; they belong to the follow-up W01 slices.  Importing them fails
  honestly instead of returning a placeholder "pass".

Nothing here launches the game, opens an asset, trains or evaluates.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "package",
]

#: Version of the *package* (the v4.3.0 development line), not of the contract.
#: The contract version is :data:`touhou_rl.v430.contracts.CONTRACT_VERSION` and
#: the plan revision it implements is
#: :data:`touhou_rl.v430.contracts.SPEC_REVISION`.
__version__ = "4.3.0"

#: Modules this package is allowed to grow, in the order the guide lists them.
#: Anything else under ``touhou_rl/v430/`` is a different work package.
PLANNED_MODULES: tuple[str, ...] = (
    "contracts",
    "actions",
    "observation",
    "session",
    "evaluate",
    "learn",
    "sequence",
    "settle",
)


def package() -> dict[str, object]:
    """Identify the package without importing any submodule.

    The dict is metadata only: it never claims that a module exists or that a
    gate passed.
    """

    return {
        "package": "touhou_rl.v430",
        "version": __version__,
        "contract_module": "touhou_rl.v430.contracts",
        "action_module": "touhou_rl.v430.actions",
        "cli": "tools/rl/v430.py",
        "planned_modules": list(PLANNED_MODULES),
    }
