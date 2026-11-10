"""WSL shadow launcher argument and Windows pipe-reader regression checks."""

from __future__ import annotations

from unittest.mock import Mock

from tools.rl import v400_wsl_env


def test_start_clears_wsl_display_and_keeps_windows_reader(monkeypatch):
    popen = Mock()
    monkeypatch.setattr(v400_wsl_env.os, "name", "nt")
    monkeypatch.setattr("subprocess.Popen", popen)
    reader = Mock()
    monkeypatch.setattr(v400_wsl_env.WslHarnessClient, "_start_pipe_reader", reader)
    client = v400_wsl_env.WslHarnessClient(
        "/tmp/th08-modern", "/tmp/game", distro="Ubuntu", timeout=2.0,
    )
    client.start()

    assert popen.call_args.args[0] == [
        "wsl.exe", "-d", "Ubuntu", "--", "env", "-u", "DISPLAY",
        "SDL_AUDIODRIVER=dummy", "/tmp/th08-modern", "--data-dir", "/tmp/game",
        "--rl-control", "0,1", "--rl-role", "shadow",
    ]
    reader.assert_called_once_with()
