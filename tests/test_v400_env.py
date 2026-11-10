"""Lifecycle contract checks for the v400 environment wrapper."""

from __future__ import annotations

import pytest

from tools.rl.v400_wsl_env import WslHarnessClient
from touhou_rl.v400.bench.reset_guard import ResetHashMismatch
from touhou_rl.v400.gym import env

#: The two hashes EXP-P0-070 run c really recorded for ``RESET 0 1 0 1 0``.
#: Test constants only: the client below is a protocol double, not an episode.
EXPECTED_RESET_HASH = "7f04c7361749ed9f"
DIVERGENT_RESET_HASH = "795b9441ed0c8056"


class _Client:
    def __init__(self, *args, **kwargs):
        self.closed = 0

    def close(self):
        self.closed += 1


class _Block:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1
        raise RuntimeError("synthetic shm close failure")


def test_close_always_closes_client_when_shared_block_close_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "HarnessClient", _Client)
    client = _Client()
    monkeypatch.setattr(env, "HarnessClient", lambda *args, **kwargs: client)
    instance = env.Th08Env(tmp_path / "harness", tmp_path, shm_name="block")
    block = _Block()
    instance._shm = block

    try:
        instance.close()
    except RuntimeError as exc:
        assert str(exc) == "synthetic shm close failure"
    else:
        raise AssertionError("shared-memory close failure should remain visible")

    assert block.closed == 1
    assert client.closed == 1
    assert instance._shm is None


def test_close_is_idempotent_after_shared_block_is_detached(monkeypatch, tmp_path):
    client = _Client()
    monkeypatch.setattr(env, "HarnessClient", lambda *args, **kwargs: client)
    instance = env.Th08Env(tmp_path / "harness", tmp_path, shm_name="block")

    instance.close()
    instance.close()

    assert client.closed == 2
    assert instance._shm is None


# -- RESET state-hash check in the WSL wrapper (EXP-P0-071) ----------------
def wsl_client(reply: str, *, expect: str | None = None) -> WslHarnessClient:
    """A WSL client whose transport is replaced by one canned RESET reply."""
    client = WslHarnessClient("/root/Touhou_RL/build/rl/th08-modern",
                              "/root/v400-data/game", expect_reset_hash=expect)
    client.request = lambda line, timeout=None: reply
    return client


def test_wsl_reset_records_the_engine_hash_without_an_expectation():
    client = wsl_client("OK reset=1 ready=1 setup_frames=330 hash=7F04C7361749ED9F")
    reply = client.reset(stage=1, difficulty=1, team=0, seed=1000)

    assert reply["hash"] == "7F04C7361749ED9F"
    assert client.reset_hashes == ["7F04C7361749ED9F"]
    assert client.reset_guard is None


def test_wsl_reset_accepts_the_configured_hash():
    client = wsl_client("OK reset=1 ready=1 hash=7f04c7361749ed9f",
                        expect="7F04C7361749ED9F")
    reply = client.reset(stage=1, difficulty=1, team=0, seed=1000)

    assert reply["hash"] == EXPECTED_RESET_HASH
    assert client.reset_guard.expected == EXPECTED_RESET_HASH


def test_wsl_reset_fails_closed_on_a_configured_hash_mismatch():
    client = wsl_client(f"OK reset=1 ready=1 hash={DIVERGENT_RESET_HASH}",
                        expect=EXPECTED_RESET_HASH)
    with pytest.raises(ResetHashMismatch) as caught:
        client.reset(stage=1, difficulty=1, team=0, seed=1000)

    message = str(caught.value)
    assert EXPECTED_RESET_HASH in message and DIVERGENT_RESET_HASH in message
    assert "RESET 1 1 0 1000 0" in message
    # The real reply is still recorded for diagnosis, but no step may follow.
    assert client.reset_hashes == [DIVERGENT_RESET_HASH]
