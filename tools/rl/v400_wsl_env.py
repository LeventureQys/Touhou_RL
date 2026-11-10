"""Minimal Windows-to-WSL launcher for the v400 line protocol.

This deliberately keeps the existing protocol unchanged: Windows owns pipes to
``wsl.exe`` and WSL owns the Linux ELF process.  Paths passed to the launcher
are Linux paths (the WSL distro can be selected explicitly).

Since EXP-P0-071 the wrapper can verify the state hash the engine itself reports
for a ``RESET``: when ``expect_reset_hash`` is configured, a reply whose ``hash``
differs raises ``ResetHashMismatch`` before any ``STEP`` is sent.  This is opt-in
so the probe and other callers keep their old behaviour, and it has no retry loop
of its own — ``tools/rl/v400_bench_run.py`` owns retries and passes its guard
into the episode instead of into this client.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from touhou_rl.v400.bench.reset_guard import (
    ResetHashGuard,
    ResetHashMismatch,
    normalize_reset_hash,
    stage_start_command,
)
from touhou_rl.v400.gym.protocol import HarnessClient


class WslHarnessClient(HarnessClient):
    """HarnessClient whose child is a real ELF launched through ``wsl.exe``."""

    def __init__(self, binary: str, data_dir: str, *, distro: str = "Ubuntu",
                 expect_reset_hash: str | None = None, **kwargs):
        self.distro = distro
        self.wsl_binary = binary
        self.wsl_data_dir = data_dir
        # HarnessClient validates Windows paths, so initialize the transport
        # state without applying host-side file validation to Linux paths.
        self.binary = Path(binary)
        self.data_dir = Path(data_dir)
        self.shm_name = kwargs.pop("shm_name", None)
        self.timeout = kwargs.pop("timeout", 300.0)
        self.stats = __import__("touhou_rl.v400.gym.protocol", fromlist=["ProtocolStats"]).ProtocolStats()
        self._proc = None
        self._buffer = b""
        self._reader_queue = None
        self._reader_thread = None
        self._env = dict(os.environ)
        self._env.update(kwargs.pop("env", {}) or {})
        if kwargs:
            raise TypeError(f"unexpected options: {', '.join(kwargs)}")
        #: Every raw ``hash`` this client's RESET replies carried, in order.
        self.reset_hashes: list[str | None] = []
        self.reset_guard = (ResetHashGuard(expected=expect_reset_hash, source="client")
                            if expect_reset_hash is not None else None)

    def start(self) -> None:
        import subprocess
        if self._proc is not None:
            raise RuntimeError("client already started")
        argv = ["wsl.exe", "-d", self.distro, "--", "env", "-u", "DISPLAY",
                "SDL_AUDIODRIVER=dummy", self.wsl_binary,
                "--data-dir", self.wsl_data_dir, "--rl-control", "0,1",
                "--rl-role", "shadow"]
        self._proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, env=self._env, bufsize=0)
        if os.name == "nt":
            self._start_pipe_reader()

    def reset(self, *, stage: int, difficulty: int, team: int, seed: int,
              practice: bool = False, timeout: float | None = None) -> dict[str, str]:
        """``RESET`` whose reply hash is kept, and checked when configured."""
        reply = super().reset(stage=stage, difficulty=difficulty, team=team, seed=seed,
                              practice=practice, timeout=timeout)
        self.reset_hashes.append(reply.get("hash"))
        if self.reset_guard is not None:
            stage_start = stage_start_command(stage=stage, difficulty=difficulty,
                                              team=team, seed=seed, practice=practice)
            self.reset_guard.check(reply, stage_start=stage_start)
        return reply


def run_probe(client: WslHarnessClient) -> dict[str, object]:
    """Exercise HELLO, RESET, STEP and QUIT and return parsed replies."""
    client.start()
    try:
        hello = client.hello()
        reset = client.reset(stage=0, difficulty=1, team=0, seed=1)
        step = client.step(0, 2)
        return {"hello": hello, "reset": reset, "step": step,
                "commands": client.stats.commands, "frames": client.stats.frames}
    finally:
        client.close()


def _reset_hash(value: str) -> str:
    try:
        return normalize_reset_hash(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--expect-reset-hash", type=_reset_hash, default=None,
                        metavar="HEX",
                        help="fail if the RESET reply hash differs from this value")
    args = parser.parse_args()
    client = WslHarnessClient(args.binary, args.data_dir, distro=args.distro,
                              expect_reset_hash=args.expect_reset_hash)
    try:
        print(run_probe(client))
    except ResetHashMismatch as exc:
        print(f"RESET state-hash guard refused the probe: {exc}")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
