"""WP5 real-engine fork checks; shadow is deliberately unavailable until null-device exists.

A Python worker or ShadowProcess is NOT a game fork. This test exercises the
compiled Linux TH08 engine when explicitly configured through environment.
"""
import os
import subprocess
from pathlib import Path

import pytest

from touhou_rl.v400.wp5 import ShadowProcess


def test_shadow_process_cannot_be_mistaken_for_real_fork():
    parent = ShadowProcess(42)
    parent.start()
    child = parent.branch()
    assert child is not parent
    assert not hasattr(child, "pid")
    assert not hasattr(child, "game_binary")
    assert child.snapshot() == parent.snapshot()
    # This is only a state-copy contract, never counted as G0.2 fork acceptance.


def test_real_game_fork_fails_closed_until_null_device_is_ready():
    binary = os.environ.get("V400_WP5_LINUX_BINARY")
    data = os.environ.get("V400_WP5_LINUX_DATA")
    if not binary or not data:
        pytest.skip("real Linux harness path not configured; shadow tests do not count as fork")
    command = [binary, "--data-dir", data, "--rl-control", "0,1", "--rl-role", "shadow"]
    if os.name == "nt":
        command = ["wsl.exe", "-d", "Ubuntu", "--", *command]
    result = subprocess.run(
        command,
        input="HELLO\nBRANCH 0 1\nPARK\nSPAWN 1\nQUIT\n", capture_output=True,
        text=True, timeout=30, check=False,
        env={**os.environ, "SDL_AUDIODRIVER": "dummy"},
    )
    combined = result.stdout + result.stderr
    # The actual null device is not yet implemented; this must be a hard reject.
    assert result.returncode != 0
    assert "shadow" in combined.lower()
    assert "branch=1" not in combined
    assert "park=1" not in combined
    assert "spawn=1" not in combined


def test_g02_acceptance_is_not_satisfied_by_shadow_worker():
    """The five G0.2 requirements have explicit real-engine evidence gates."""
    required = (
        "5000-frame-hash", "20-branch-300-frame", "10000-frame-shadow",
        "trajectory-pixel-byte", "fork-and-branch-frame-timing",
    )
    evidence = os.environ.get("V400_WP5_REAL_EVIDENCE")
    if not evidence:
        pytest.skip("G0.2 real-engine evidence missing: " + ", ".join(required))
    document = Path(evidence).read_text(encoding="utf-8")
    assert all(name in document for name in required)
    assert "real_game_fork_verified=true" in document
