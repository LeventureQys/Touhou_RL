"""Synthetic accounting/fail-closed tests; NEVER real throughput evidence."""
import pytest

from tools.rl.v400_shadow_throughput import Unavailable
from tools.rl.v400_shadow_throughput_stats import SeparateProcessClient, measure_samples


class Client:
    def __init__(self):
        self.role = "shadow"
        self.threads = "1"
        self.subthread = "0"
        self.state = "2"
        self.frame = 0
        self.resets = []
        self.calls = 0
        self.ready = "1"
        self.chunk = 100
        self.exit_at = None
        self.invalid_at = None

    def hello(self):
        return {"role": self.role}

    def debug(self):
        return {"threads": self.threads, "subthread": self.subthread,
                "curState": self.state, "frames": str(self.frame)}

    def reset(self, **kwargs):
        self.resets.append(kwargs)
        self.frame = 0
        self.state = "2"
        return {"ready": self.ready}

    def step(self, mask, frames):
        self.calls += 1
        if self.calls == self.invalid_at:
            return {"advanced": "bad", "exit": "0"}
        count = min(frames, self.chunk)
        self.frame += count
        if self.calls == self.exit_at:
            self.state = "6"
        return {"advanced": str(count), "exit": "0"}


def ticks(samples=3):
    values = iter(float(index) for index in range(samples * 20))
    return lambda: next(values)


def test_process_restarts_between_resets():
    instances = []

    class Process(Client):
        def __init__(self, binary, data_dir, *, timeout):
            super().__init__()
            self.closed = False
            instances.append(self)

        def start(self):
            pass

        def close(self):
            self.closed = True

        def reset(self, **kwargs):
            if self.resets:
                return {"ready": "0"}
            return super().reset(**kwargs)

    with SeparateProcessClient("binary", "data", factory=Process) as client:
        result = measure_samples(client, clock=ticks())
    assert result["sample_count"] == len(instances) == 3
    assert all(instance.closed and len(instance.resets) == 1 for instance in instances)



def test_three_fresh_reset_bounded_samples_and_gate():
    client = Client()
    result = measure_samples(client, clock=ticks())
    assert result["sample_count"] == 3
    assert result["total_measured_frames"] == 3000
    assert len(client.resets) == 3
    assert all(reset == {"stage": 0, "difficulty": 1, "team": 0,
                         "seed": 1, "practice": True} for reset in client.resets)
    assert [row["debug_before"]["frames"] for row in result["samples"]] == ["0"] * 3
    assert [row["debug_after"]["frames"] for row in result["samples"]] == ["1000"] * 3
    assert result["mean_frames_per_second_per_core"] == 100.0
    assert result["interval_min_max_frames_per_second_per_core"] == [100.0, 100.0]
    assert result["core_gate_pass_all_samples"] is False


@pytest.mark.parametrize("samples,frames", [(2, 1000), (3, 999), (3, 1001)])
def test_rejects_insufficient_or_different_samples(samples, frames):
    with pytest.raises(Unavailable, match="at least 3 samples of exactly 1000"):
        measure_samples(Client(), samples=samples, frames=frames)


@pytest.mark.parametrize("attribute,value,matching", [
    ("role", "env", "HELLO"), ("threads", "2", "DEBUG"),
    ("subthread", "1", "DEBUG"), ("ready", "0", "RESET"),
])
def test_rejects_invalid_setup(attribute, value, matching):
    client = Client()
    setattr(client, attribute, value)
    with pytest.raises(Unavailable, match=matching):
        measure_samples(client)
    assert client.calls == 0


@pytest.mark.parametrize("bad_at", [1, 11, 30])
def test_rejects_crossing_curstate6_at_any_sample(bad_at):
    client = Client()
    client.exit_at = bad_at
    with pytest.raises(Unavailable, match="curState2"):
        measure_samples(client, clock=ticks())


def test_rejects_bad_step_and_frame_accounting():
    client = Client()
    client.invalid_at = 12
    with pytest.raises(Unavailable, match="invalid STEP advanced"):
        measure_samples(client, clock=ticks())
    client = Client()
    original = client.step

    def inconsistent(mask, frames):
        reply = original(mask, frames)
        client.frame -= 1
        return reply

    client.step = inconsistent
    with pytest.raises(Unavailable, match="frame mismatch"):
        measure_samples(client, clock=ticks())


def test_rejects_invalid_step_time():
    with pytest.raises(Unavailable, match="STEP wall time"):
        measure_samples(Client(), clock=lambda: 0.0)
