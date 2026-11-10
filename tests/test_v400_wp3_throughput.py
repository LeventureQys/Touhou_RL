"""WP2 throughput accounting tests; fake clients do not prove engine performance."""

import pytest

from touhou_rl.v400.throughput import ShadowUnavailable, measure_shadow


class Client:
    def __init__(self, role="shadow", threads="1", subthread="0", advanced=100):
        self.role, self.threads, self.subthread, self.advanced = role, threads, subthread, advanced
        self.steps = []

    def hello(self):
        return {"role": self.role}

    def debug(self):
        return {"threads": self.threads, "subthread": self.subthread}

    def step(self, mask, frames):
        self.steps.append((mask, frames))
        return {"advanced": str(min(frames, self.advanced))}


def test_rejects_gl_env_even_with_readback_disabled():
    client = Client(role="env")
    with pytest.raises(ShadowUnavailable, match="engine role"):
        measure_shadow(client)
    assert not client.steps


@pytest.mark.parametrize("threads,subthread", [("2", "0"), ("1", "1")])
def test_rejects_non_quiescent_shadow(threads, subthread):
    with pytest.raises(ShadowUnavailable, match="not quiescent"):
        measure_shadow(Client(threads=threads, subthread=subthread))


def test_accounts_all_frames_and_elapsed_time():
    ticks = iter((5.0, 7.0))
    client = Client(advanced=40)
    result = measure_shadow(client, frames=100, clock=lambda: next(ticks))
    assert result == {"kind": "real-shadow", "frames": 100, "seconds": 2,
                      "frames_per_second": 50, "threads_before": 1, "threads_after": 1}
    assert client.steps == [(0, 100), (0, 60), (0, 20)]


def test_rejects_early_game_exit():
    class Exits(Client):
        def step(self, mask, frames):
            return {"advanced": "10", "exit": "1"}

    with pytest.raises(ShadowUnavailable, match="exited"):
        measure_shadow(Exits(), frames=100)
