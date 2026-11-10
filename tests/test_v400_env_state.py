"""Privilege contract for ``Th08Env.state()`` (03 §5, 20 §7).

Regression guard for the T0 finding: ``obs_spec`` used to swallow the ``state()``
definition, so the method body sat *after* ``obs_spec``'s ``return`` and could
never run -- and, because ``def`` never executed, ``Th08Env.state`` did not exist
at all.  Anything holding a privileged backend (``touhou_rl.v400.rl.env``) then
failed with ``AttributeError`` instead of returning the 25 tracked fields.

Nothing here starts an engine: the shared block and the protocol client are
doubles, which is exactly what the behaviour under test (read a block, or refuse
to) needs.
"""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl.v400.gym import env, shm


class _Block:
    """Stand-in for ``SharedBlock``: fixed header, 25 tracked fields, no mmap."""

    def __init__(self, fields) -> None:
        self._fields = np.asarray(fields, dtype="<u4")
        self.closed = 0

    def header(self) -> dict[str, int]:
        return {
            "version": shm.VERSION,
            "frame": 7,
            "state_hash": 0,
            "width": 640,
            "height": 480,
            "state_bytes": shm.STATE_BYTES,
            "flags": shm.FLAG_STATE_VALID | shm.FLAG_PIXELS_VALID,
        }

    def state(self) -> np.ndarray:
        return self._fields

    def pixels(self) -> np.ndarray:
        return np.zeros((480, 640, 4), dtype=np.uint8)

    def close(self) -> None:
        self.closed += 1


class _Client:
    """Protocol double: records ``close`` and never talks to a process."""

    def __init__(self, *args, **kwargs) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _environment(monkeypatch, tmp_path, *, privileged: bool) -> tuple[env.Th08Env, _Block]:
    """An attached ``Th08Env`` whose block holds ``100 + index`` in every field."""
    monkeypatch.setattr(env, "HarnessClient", _Client)
    instance = env.Th08Env(tmp_path / "harness", tmp_path, shm_name="block", privileged=privileged)
    block = _Block([100 + index for index in range(shm.STATE_FIELDS)])
    instance._shm = block
    return instance, block


# -- the regression itself ------------------------------------------------
def test_state_is_a_method_not_part_of_the_obs_spec_property():
    """``state`` must be its own function; swallowing it is the bug being fixed."""
    assert "state" in vars(env.Th08Env), "Th08Env.state is missing: obs_spec swallowed it again"
    assert not isinstance(vars(env.Th08Env)["state"], property)
    assert callable(env.Th08Env.state)


def test_obs_spec_still_describes_the_rgba_observation(monkeypatch, tmp_path):
    instance, _ = _environment(monkeypatch, tmp_path, privileged=True)

    spec = instance.obs_spec

    assert spec["shape"] == (480, 640, 4)
    assert spec["dtype"] is np.uint8
    assert (spec["low"], spec["high"]) == (0, 255)


# -- privileged mode ------------------------------------------------------
def test_privileged_state_returns_all_tracked_fields(monkeypatch, tmp_path):
    instance, _ = _environment(monkeypatch, tmp_path, privileged=True)

    fields = instance.state()

    assert len(fields) == shm.STATE_FIELDS == 25
    assert list(fields) == list(range(shm.STATE_FIELDS))
    assert fields == {index: 100 + index for index in range(shm.STATE_FIELDS)}
    assert all(isinstance(value, int) for value in fields.values())


def test_privileged_state_is_read_from_the_attached_block_each_call(monkeypatch, tmp_path):
    instance, block = _environment(monkeypatch, tmp_path, privileged=True)
    assert instance.state()[env.F_DEATHS] == 100 + env.F_DEATHS

    block._fields = np.asarray([200 + index for index in range(shm.STATE_FIELDS)], dtype="<u4")

    assert instance.state()[env.F_DEATHS] == 200 + env.F_DEATHS


def test_privileged_info_carries_the_state(monkeypatch, tmp_path):
    instance, _ = _environment(monkeypatch, tmp_path, privileged=True)

    info = instance._info(events=["death"])

    assert info.privileged is True
    assert info.frame == 7
    assert info.state == instance.state()


# -- zero-privilege mode --------------------------------------------------
def test_zero_privilege_state_raises_privilege_error(monkeypatch, tmp_path):
    instance, _ = _environment(monkeypatch, tmp_path, privileged=False)

    with pytest.raises(env.PrivilegeError) as caught:
        instance.state()

    assert "zero-privilege" in str(caught.value)


def test_zero_privilege_info_has_no_state(monkeypatch, tmp_path):
    instance, _ = _environment(monkeypatch, tmp_path, privileged=False)

    info = instance._info(events=["bomb"])

    assert info.privileged is False
    assert info.state is None


def test_zero_privilege_debug_also_refuses(monkeypatch, tmp_path):
    instance, _ = _environment(monkeypatch, tmp_path, privileged=False)

    with pytest.raises(env.PrivilegeError):
        instance.debug()
