"""Minimal vector-environment contract checks for v400."""

from __future__ import annotations

import time

import pytest

from touhou_rl.v400.gym import VecTh08Env


class _FakeEnv:
    def __init__(self, name: str, *, close_error: BaseException | None = None, delay: float = 0.0) -> None:
        self.name = name
        self.close_error = close_error
        self.delay = delay
        self.reset_calls = 0
        self.actions = []
        self.closed = 0

    def reset(self):
        self.reset_calls += 1
        if self.delay:
            time.sleep(self.delay)
        return f"reset-{self.name}"

    def step(self, action):
        self.actions.append(action)
        if self.delay:
            time.sleep(self.delay)
        return (f"obs-{self.name}", action, False, False, {"env": self.name})

    def close(self):
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


def test_reset_and_step_dispatch_in_order():
    envs = [_FakeEnv("a"), _FakeEnv("b")]
    vector = VecTh08Env(envs)

    assert vector.reset() == ["reset-a", "reset-b"]
    assert vector.step([3, 7]) == [
        ("obs-a", 3, False, False, {"env": "a"}),
        ("obs-b", 7, False, False, {"env": "b"}),
    ]
    assert [env.reset_calls for env in envs] == [1, 1]
    assert [env.actions for env in envs] == [[3], [7]]


def test_step_runs_environments_concurrently():
    vector = VecTh08Env([_FakeEnv("a", delay=0.08), _FakeEnv("b", delay=0.08)])
    started = time.perf_counter()
    vector.step([1, 2])
    elapsed = time.perf_counter() - started
    assert elapsed < 0.14


    vector = VecTh08Env([_FakeEnv("a"), _FakeEnv("b")])

    with pytest.raises(ValueError, match="one item per environment"):
        vector.step([1])


def test_close_continues_after_first_error_and_reraises_it():
    first_error = RuntimeError("first close failure")
    envs = [
        _FakeEnv("a", close_error=first_error),
        _FakeEnv("b", close_error=ValueError("second close failure")),
        _FakeEnv("c"),
    ]

    with pytest.raises(RuntimeError, match="first close failure"):
        VecTh08Env(envs).close()

    assert [env.closed for env in envs] == [1, 1, 1]


def test_empty_environment_sequence_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        VecTh08Env([])
