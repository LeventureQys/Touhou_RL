"""Observation metadata contract checks for the v400 environment wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl.v400.gym import env


class _FakeClient:
    def close(self):
        pass


class _FakeShm:
    def pixels(self):
        return np.zeros((3, 5, 4), dtype=np.uint8)


def test_obs_spec_describes_rgba8_pixels(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "HarnessClient", lambda *args, **kwargs: _FakeClient())
    instance = env.Th08Env(tmp_path / "harness", tmp_path, shm_name="block")
    instance._shm = _FakeShm()

    spec = instance.obs_spec

    assert spec == {
        "shape": (3, 5, 4),
        "dtype": np.uint8,
        "low": 0,
        "high": 255,
    }


def test_obs_spec_requires_an_attached_shared_block(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "HarnessClient", lambda *args, **kwargs: _FakeClient())
    instance = env.Th08Env(tmp_path / "harness", tmp_path, shm_name="block")

    with pytest.raises(RuntimeError, match="context manager"):
        _ = instance.obs_spec


def test_obs_spec_is_read_only(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "HarnessClient", lambda *args, **kwargs: _FakeClient())
    instance = env.Th08Env(tmp_path / "harness", tmp_path, shm_name="block")
    instance._shm = _FakeShm()

    with pytest.raises(AttributeError):
        instance.obs_spec = {}
