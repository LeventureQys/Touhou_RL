"""Measure real WSL TH08 one-thread null-device (shadow) STEP throughput.

Run *inside WSL* with the real Linux binary and installed game data. Exit 3
means unavailable; neither a GL path nor LP_NO_RAST is accepted as a proxy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from touhou_rl.v400.gym.protocol import HarnessClient, HarnessError


class Unavailable(RuntimeError):
    """Real shadow proof or requested logical-frame accounting failed."""


class ShadowClient(HarnessClient):
    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("client already started")
        env = dict(self._env)
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        env.pop("LP_NO_RAST", None)
        self._proc = subprocess.Popen(
            [str(self.binary), "--data-dir", str(self.data_dir), "--rl-control", "0,1",
             "--rl-role", "shadow"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, bufsize=0,
        )


def verify(client: HarnessClient) -> dict[str, str]:
    hello = client.hello()
    if hello.get("role") != "shadow":
        raise Unavailable(f"HELLO role={hello.get('role')!r}; require shadow")
    debug = client.debug()
    if debug.get("threads") != "1" or debug.get("subthread") != "0":
        raise Unavailable(f"DEBUG threads={debug.get('threads')!r} subthread={debug.get('subthread')!r}; require 1/0")
    return debug


def measure(client: HarnessClient, frames: int, *, clock=time.perf_counter) -> dict:
    if frames < 1:
        raise Unavailable("frames must be positive")
    initial = verify(client)
    ready = client.reset(stage=0, difficulty=1, team=0, seed=1, practice=True)
    if ready.get("ready") != "1":
        raise Unavailable(f"RESET ready={ready.get('ready')!r}; require 1")
    before = verify(client)
    begin = clock()
    advanced = 0
    steps = 0
    while advanced < frames:
        requested = min(frames - advanced, 100)
        response = client.step(0, requested)
        try:
            n = int(response.get("advanced", ""))
        except ValueError as exc:
            raise Unavailable(f"STEP invalid advanced: {response}") from exc
        if n < 1 or n > requested:
            raise Unavailable(f"STEP did not advance valid frames: {response}")
        advanced += n
        steps += 1
        if response.get("exit") == "1":
            raise Unavailable(f"STEP game exited after {advanced}/{frames} frames")
    seconds = clock() - begin
    after = verify(client)
    if before.get("curState") != "2" or after.get("curState") != "2":
        raise Unavailable(f"STEP left active gameplay: state before={before.get('curState')} after={after.get('curState')}; advanced={advanced}/{frames}")
    if seconds <= 0:
        raise Unavailable("non-positive STEP wall time")
    return {"status": "measured", "kind": "real-shadow", "frames": advanced,
            "wall_seconds": seconds, "frames_per_second_per_core": advanced / seconds,
            "steps": steps, "hello_role": "shadow", "reset_ready": ready["ready"],
            "debug_initial": initial, "debug_before": before, "debug_after": after,
            "logical_cores_used": 1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        if os.name != "posix" or "microsoft" not in Path("/proc/version").read_text().lower():
            raise Unavailable("requires real WSL Linux binary/data")
        if "LP_NO_RAST" in os.environ:
            raise Unavailable("LP_NO_RAST proxy is forbidden; unset it")
        with ShadowClient(args.binary, args.data_dir, timeout=120) as client:
            result = measure(client, args.frames)
        result.update(binary=str(args.binary.resolve()), data_dir=str(args.data_dir.resolve()))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (HarnessError, Unavailable, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "unavailable", "reason": str(exc),
                          "measured_frames_per_second_per_core": None}, ensure_ascii=False))
        return 3


if __name__ == "__main__":
    sys.exit(main())
