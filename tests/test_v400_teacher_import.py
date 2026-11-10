"""Top-level import contract for ``touhou_rl.v400.teacher`` (regression test).

The package once failed at ``import touhou_rl.v400.teacher``:

``hazard_tape.py`` imported ``SpeedTable`` from ``geometry``, but the class only
existed in ``motion.py``, so every consumer of the teacher (including the P1
search CLI) died on an ``ImportError`` before it could run.

This module pins three things:

* the package imports and every name in ``__all__`` is actually reachable;
* the four-speed table has **one** definition, shared by ``geometry``,
  ``motion`` and ``hazard_tape`` (no second class with the same fields);
* that class can be built both from measured numbers and from a ``PLAYER``
  reply, and refuses a reply that is missing a speed instead of guessing zero.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

import touhou_rl.v400.teacher as teacher


def test_import_succeeds_and_every_export_is_reachable():
    """The regression: the bare import above must not raise."""

    assert teacher.__all__, "teacher.__all__ is empty"
    assert len(set(teacher.__all__)) == len(teacher.__all__), "duplicate export"
    for name in teacher.__all__:
        assert hasattr(teacher, name), f"teacher.__all__ names {name!r}, but it is missing"
        assert getattr(teacher, name) is not None


def test_exported_callables_are_callable():
    for name in teacher.__all__:
        value = getattr(teacher, name)
        if isinstance(value, type) or callable(value):
            assert callable(value), name


def test_speed_table_has_exactly_one_definition():
    geometry = importlib.import_module("touhou_rl.v400.teacher.geometry")
    motion = importlib.import_module("touhou_rl.v400.teacher.motion")
    hazard_tape = importlib.import_module("touhou_rl.v400.teacher.hazard_tape")

    assert geometry.SpeedTable is motion.SpeedTable
    assert hazard_tape.SpeedTable is geometry.SpeedTable


def test_speed_table_is_constructible_from_measured_values():
    geometry = importlib.import_module("touhou_rl.v400.teacher.geometry")
    table = geometry.SpeedTable(4.0, 2.828427, 2.0, 1.414214)

    assert table.as_tuple() == (4.0, 2.828427, 2.0, 1.414214)
    assert table.axial == 4.0
    assert table.diagonal == 2.828427
    assert table.focused_axial == 2.0
    assert table.focused_diagonal == 1.414214
    table.check()


def test_speed_table_is_constructible_from_a_player_reply():
    geometry = importlib.import_module("touhou_rl.v400.teacher.geometry")
    player = SimpleNamespace(
        speed_normal_axial=4.0,
        speed_normal_diagonal=2.828427,
        speed_focused_axial=2.0,
        speed_focused_diagonal=1.414214,
    )

    assert geometry.SpeedTable.from_player(player).as_tuple() == (
        4.0,
        2.828427,
        2.0,
        1.414214,
    )


def test_speed_table_refuses_a_reply_missing_a_speed():
    geometry = importlib.import_module("touhou_rl.v400.teacher.geometry")
    player = SimpleNamespace(
        speed_normal_axial=4.0,
        speed_normal_diagonal=2.828427,
        speed_focused_axial=2.0,
        speed_focused_diagonal=None,
    )

    with pytest.raises(ValueError, match="speed_focused_diagonal"):
        geometry.SpeedTable.from_player(player)


def test_speed_table_check_rejects_non_positive_speed():
    geometry = importlib.import_module("touhou_rl.v400.teacher.geometry")

    with pytest.raises(ValueError, match="normal_axial"):
        geometry.SpeedTable(0.0, 2.828427, 2.0, 1.414214).check()


def test_import_is_independent_of_the_current_working_directory(tmp_path):
    """A fresh interpreter launched elsewhere must import the same package."""

    code = (
        "import touhou_rl.v400.teacher as t; "
        "assert all(hasattr(t, n) for n in t.__all__); "
        "print(len(t.__all__))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert completed.stdout.strip() == str(len(teacher.__all__))
