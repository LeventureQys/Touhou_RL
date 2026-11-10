"""Accounting tests are synthetic only; real WSL invocation supplies performance evidence."""
import pytest

from tools.rl.v400_shadow_throughput import Unavailable, measure


class Client:
    def __init__(self, role="shadow", threads="1", subthread="0", ready="1", chunk=100):
        self.role, self.threads, self.subthread, self.ready, self.chunk = role, threads, subthread, ready, chunk
        self.steps = []

    def hello(self):
        return {"role": self.role}

    def debug(self):
        return {"threads": self.threads, "subthread": self.subthread, "curState": "2"}

    def reset(self, **kwargs):
        self.reset_args = kwargs
        return {"ready": self.ready}

    def step(self, mask, frames):
        self.steps.append((mask, frames))
        return {"advanced": str(min(frames, self.chunk)), "exit": "0"}


@pytest.mark.parametrize("settings,matching", [
    ({"role": "env"}, "HELLO"), ({"threads": "2"}, "DEBUG"),
    ({"subthread": "1"}, "DEBUG"), ({"ready": "0"}, "RESET"),
])
def test_fails_closed_before_step(settings, matching):
    client = Client(**settings)
    with pytest.raises(Unavailable, match=matching):
        measure(client, 100)
    assert not client.steps


def test_counts_only_advanced_frames_and_step_wall_time():
    client = Client(chunk=40)
    ticks = iter([5.0, 7.0])
    result = measure(client, 100, clock=lambda: next(ticks))
    assert result["frames"] == 100
    assert result["wall_seconds"] == 2.0
    assert result["frames_per_second_per_core"] == 50.0
    assert result["logical_cores_used"] == 1
    assert client.steps == [(0, 100), (0, 60), (0, 20)]
    assert client.reset_args["practice"] is True


@pytest.mark.parametrize("advanced", ["0", "bad", "101"])
def test_rejects_invalid_step_accounting(advanced):
    client = Client()
    client.step = lambda mask, frames: {"advanced": advanced}
    with pytest.raises(Unavailable, match="STEP"):
        measure(client, 100)


def test_rejects_early_exit():
    client = Client()
    client.step = lambda mask, frames: {"advanced": "10", "exit": "1"}
    with pytest.raises(Unavailable, match="exited"):
        measure(client, 100)
