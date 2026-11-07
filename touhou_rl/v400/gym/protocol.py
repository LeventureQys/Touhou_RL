"""Control-protocol client for the TH08 harness (03 §4.2, 10-P0 §3).

The engine speaks a line-oriented ASCII protocol over a pipe pair: one request
per line, one reply per line.  Bulk data does not travel here — it lands in the
shared block (see :mod:`touhou_rl.v400.gym.shm`).

Starting the process is deliberately explicit: this object owns the child's
lifetime and must be closed, so it doubles as a context manager.
"""

from __future__ import annotations

import os
import select
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT = 300.0


class HarnessError(RuntimeError):
    """The engine replied with ``ERR`` or died."""


@dataclass
class ProtocolStats:
    """Cheap accounting so the budget ledger can be filled from real usage."""

    commands: int = 0
    frames: int = 0
    wall_seconds: float = 0.0
    replies: dict[str, int] = field(default_factory=dict)


class HarnessClient:
    """Owns one ``th08-modern --rl-control`` process."""

    def __init__(
        self,
        binary: str | os.PathLike[str],
        data_dir: str | os.PathLike[str],
        *,
        shm_name: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.binary = Path(binary).expanduser()
        self.data_dir = Path(data_dir).expanduser()
        if not self.binary.is_file():
            raise FileNotFoundError(f"harness binary not found: {self.binary}")
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"game data dir not found: {self.data_dir}")

        self.shm_name = shm_name
        self.timeout = timeout
        self.stats = ProtocolStats()
        self._proc: subprocess.Popen[bytes] | None = None
        self._buffer = b""
        self._env = dict(os.environ)
        # Software rendering keeps a single instance reproducible; the caller can
        # override for the GPU backend once that path is healthy again.
        self._env.update(
            SDL_AUDIODRIVER="dummy",
            LIBGL_ALWAYS_SOFTWARE="1",
            GALLIUM_DRIVER="llvmpipe",
            LP_NUM_THREADS="1",
        )
        if env:
            self._env.update(env)

    # -- lifetime ---------------------------------------------------------
    def __enter__(self) -> "HarnessClient":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("client already started")
        argv = [str(self.binary), "--data-dir", str(self.data_dir), "--rl-control", "0,1"]
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            bufsize=0,
        )

    def close(self, *, quit_timeout: float = 30.0) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        try:
            if proc.poll() is None:
                try:
                    self._write("QUIT")
                    self._readline(quit_timeout)
                except Exception:
                    pass
            try:
                proc.wait(timeout=quit_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass

    # -- transport --------------------------------------------------------
    def _write(self, line: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((line + "\n").encode())
        self._proc.stdin.flush()

    def _readline(self, timeout: float) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self._proc.stdout], [], [], 0.25)
            if not ready:
                if self._proc.poll() is not None:
                    raise HarnessError(f"harness exited before replying (code {self._proc.returncode})")
                continue
            chunk = os.read(self._proc.stdout.fileno(), 8192)
            if not chunk:
                raise HarnessError("harness closed the control pipe")
            self._buffer += chunk
            if b"\n" in self._buffer:
                line, _, self._buffer = self._buffer.partition(b"\n")
                return line.decode("utf-8", "replace")
        raise HarnessError(f"no reply within {timeout:.1f}s")

    def request(self, line: str, timeout: float | None = None) -> str:
        """Send one command, return one reply, raise on ``ERR``."""
        if self._proc is None:
            raise RuntimeError("client is not running")
        started = time.monotonic()
        self._write(line)
        reply = self._readline(self.timeout if timeout is None else timeout)
        self.stats.commands += 1
        self.stats.wall_seconds += time.monotonic() - started
        verb = reply.split(" ", 1)[0]
        self.stats.replies[verb] = self.stats.replies.get(verb, 0) + 1
        if reply.startswith("ERR"):
            raise HarnessError(f"{line!r} -> {reply}")
        return reply

    # -- typed commands ---------------------------------------------------
    def hello(self) -> dict[str, str]:
        return _parse(self.request("HELLO"))

    def frame(self) -> int:
        return int(_parse(self.request("FRAME"))["frame"])

    def state_hash(self) -> tuple[str, int]:
        fields = _parse(self.request("HASH"))
        return fields["hash"], int(fields["frames"])

    def state(self) -> dict[str, int]:
        """The 25 tracked fields as ``{"f0": .., "f24": ..}``."""
        fields = _parse(self.request("STATE"))
        return {key: int(value) for key, value in fields.items() if key.startswith("f")}

    def debug(self) -> dict[str, str]:
        return _parse(self.request("DEBUG"))

    def attach_shm(self, name: str | None = None) -> dict[str, str]:
        name = name or self.shm_name
        if not name:
            raise ValueError("no shared-memory name given")
        reply = _parse(self.request(f"SHM /{name.lstrip('/')}"))
        self.shm_name = name.lstrip("/")
        return reply

    def reset(
        self,
        *,
        stage: int,
        difficulty: int,
        team: int,
        seed: int,
        practice: bool = False,
        timeout: float | None = None,
    ) -> dict[str, str]:
        cmd = f"RESET {stage} {difficulty} {team} {seed} {1 if practice else 0}"
        return _parse(self.request(cmd, timeout=timeout))

    def step(self, mask: int, frames: int) -> dict[str, str]:
        if not 0 <= mask <= 0xFFFF:
            raise ValueError(f"mask out of range: {mask}")
        if frames <= 0:
            raise ValueError(f"frames must be positive, got {frames}")
        fields = _parse(self.request(f"STEP {mask} {frames}"))
        self.stats.frames += int(fields.get("advanced", 0))
        return fields

    def readback(self, enabled: bool | None = None) -> dict[str, str]:
        if enabled is None:
            return _parse(self.request("READBACK"))
        return _parse(self.request(f"READBACK {1 if enabled else 0}"))

    def pixels(self, path: str | None = None) -> dict[str, str]:
        return _parse(self.request(f"PIXELS {path}" if path else "PIXELS"))


def _parse(reply: str) -> dict[str, str]:
    """``OK a=1 b=2`` -> ``{"_verb": "OK", "a": "1", "b": "2"}``."""
    parts = reply.split()
    out: dict[str, str] = {"_verb": parts[0] if parts else ""}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out
