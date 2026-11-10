"""Explicit recovery lifecycle checks for the v400 environment wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl.v400.gym import env


class _Client:
    instances = []
    close_error = None

    def __init__(self, *args, **kwargs):
        self.shm_name = kwargs.get("shm_name", "block")
        self.calls = []
        self.closed = False
        type(self).instances.append(self)

    def start(self):
        self.calls.append("start")

    def hello(self):
        self.calls.append("hello")

    def attach_shm(self):
        self.calls.append("attach_shm")

    def readback(self, enabled):
        self.calls.append(("readback", enabled))

    def close(self):
        self.calls.append("close")
        self.closed = True
        if type(self).close_error is not None:
            error = type(self).close_error
            type(self).close_error = None
            raise error

    def reset(self, **kwargs):
        self.calls.append(("reset", kwargs))
        return {"ready": "1"}


class _Shm:
    instances = []

    def __init__(self, name):
        self.name = name
        self.closed = False
        type(self).instances.append(self)

    def close(self):
        self.closed = True

    def pixels(self):
        return np.zeros((1, 1, 4), dtype=np.uint8)

    def header(self):
        return {"frame": 0}

    def state(self):
        return np.zeros(25, dtype=np.uint32)


def _install(monkeypatch):
    _Client.instances = []
    _Shm.instances = []
    monkeypatch.setattr(env, "HarnessClient", _Client)
    monkeypatch.setattr(env, "SharedBlock", _Shm)


def test_recover_recreates_client_and_repeats_handshake(monkeypatch, tmp_path):
    _install(monkeypatch)
    instance = env.Th08Env(tmp_path / "harness", tmp_path, shm_name="block")

    instance.__enter__()
    old_client = instance._client
    old_shm = instance.shm
    instance.recover()

    assert old_client.closed
    assert old_shm.closed
    assert instance._client is not old_client
    assert instance.shm is not old_shm
    assert instance._client.calls == ["start", "hello", "attach_shm", ("readback", True)]


def test_recover_continues_when_old_resource_close_fails(monkeypatch, tmp_path):
    _install(monkeypatch)
    instance = env.Th08Env(tmp_path / "harness", tmp_path, shm_name="block")
    instance.__enter__()
    old_client = instance._client
    old_shm = instance.shm

    def bad_close():
        old_shm.closed = True
        raise RuntimeError("synthetic shm close failure")

    old_shm.close = bad_close
    _Client.close_error = RuntimeError("synthetic client close failure")

    with pytest.raises(RuntimeError, match="synthetic shm close failure"):
        instance.recover()

    assert old_shm.closed
    assert old_client.closed
    assert instance._client is not old_client
    assert instance.shm is not old_shm
    assert instance._client.calls[:3] == ["start", "hello", "attach_shm"]


def test_recover_allows_explicit_reset_afterward(monkeypatch, tmp_path):
    _install(monkeypatch)
    instance = env.Th08Env(tmp_path / "harness", tmp_path, shm_name="block")
    instance.__enter__()
    instance.recover()

    observation, info = instance.reset(env.StageStart(stage=1, difficulty="easy", team=0, seed=7))

    assert observation.shape == (1, 1, 4)
    assert info.events == ["reset"]
    assert instance._client.calls[-1][0] == "reset"
