"""Strict WP2 null-device throughput probe; no GL/LP_NO_RAST proxy accepted.

A measurement is valid only if the engine itself reports role=shadow and one
thread both before and after the sample.  This module deliberately cannot turn
an env-role process into a shadow by changing readback settings.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class ShadowUnavailable(RuntimeError):
    """Engine has not demonstrated a one-thread null-device shadow role."""


def _require_shadow(client: object) -> dict[str, str]:
    hello = client.hello()
    if hello.get("role") != "shadow":
        raise ShadowUnavailable(f"engine role is {hello.get('role')!r}, expected shadow")
    debug = client.debug()
    if debug.get("threads") != "1" or debug.get("subthread") != "0":
        raise ShadowUnavailable(
            f"shadow not quiescent: threads={debug.get('threads')!r} "
            f"subthread={debug.get('subthread')!r}"
        )
    return debug


def measure_shadow(client: object, *, frames: int = 1000,
                   clock: Callable[[], float] = time.perf_counter) -> dict[str, object]:
    """Measure STEP throughput after independently checking the engine role.

    The caller must start and RESET the real engine; synthetic clients in unit
    tests validate accounting only and are never evidence of engine throughput.
    """
    if frames < 1:
        raise ValueError("frames must be positive")
    before = _require_shadow(client)
    begin = clock()
    remaining = frames
    advanced = 0
    while remaining:
        reply = client.step(0, min(remaining, 100000))
        n = int(reply.get("advanced", "0"))
        if n <= 0 or n > remaining:
            raise ShadowUnavailable(f"engine did not advance requested frames: {reply}")
        advanced += n
        remaining -= n
        if reply.get("exit") == "1" and remaining:
            raise ShadowUnavailable(f"engine exited after {advanced}/{frames} frames")
    elapsed = clock() - begin
    after = _require_shadow(client)
    if elapsed <= 0:
        raise ShadowUnavailable("non-positive elapsed time")
    return {"kind": "real-shadow", "frames": advanced, "seconds": elapsed,
            "frames_per_second": advanced / elapsed, "threads_before": int(before["threads"]),
            "threads_after": int(after["threads"])}
