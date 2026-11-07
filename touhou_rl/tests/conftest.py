"""Shared pytest configuration for this tree.

Only registers the ``harness`` marker; it deliberately does not add fixtures or
hooks, so it cannot change the behaviour of the existing tests (30 §2 asks for
this file to be introduced carefully for exactly that reason).

Tests that need the compiled training ground carry ``@pytest.mark.harness`` and
skip when the binary is missing, rather than failing a machine that has not built
it yet (30 §2).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Where the harness build is expected. Override with V400_HARNESS_BIN.
DEFAULT_HARNESS_BIN = Path.home() / "Touhou_RL/third_party/th08-rl/build/rl/th08-modern"
#: Game assets. Override with V400_DATA (matches tools/rl/v400_env_check.py).
DEFAULT_DATA_DIR = Path(os.environ.get("V400_DATA", str(Path.home() / "v400-data"))) / "game"


def harness_binary() -> Path:
    return Path(os.environ.get("V400_HARNESS_BIN", str(DEFAULT_HARNESS_BIN)))


def game_data_dir() -> Path:
    return Path(os.environ.get("V400_GAME_DIR", str(DEFAULT_DATA_DIR)))


def harness_available() -> bool:
    binary = harness_binary()
    data = game_data_dir()
    return binary.is_file() and os.access(binary, os.X_OK) and (data / "th08.dat").is_file()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "harness: needs the compiled TH08_RL_HARNESS binary and game data; "
        "skipped when they are absent",
    )


@pytest.fixture
def harness_paths():
    """``(binary, data_dir)`` or skip."""
    if not harness_available():
        pytest.skip(
            f"harness binary or game data missing "
            f"(binary={harness_binary()}, data={game_data_dir()})"
        )
    return harness_binary(), game_data_dir()
