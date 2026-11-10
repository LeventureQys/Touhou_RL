"""Real WSL shadow throughput: separate 1,000-frame RESET-bounded samples.

No synthetic or raster proxy is accepted. Exit 3 means unavailable and never
promotes partial/cross-curState6 samples into throughput evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from tools.rl.v400_shadow_throughput import ShadowClient, Unavailable, verify
from touhou_rl.v400.gym.protocol import HarnessError

CORE_GATE_FPS = 5000.0


class SeparateProcessClient:
    """One real shadow process per sample; RESET is only valid at startup."""

    def __init__(self, binary: Path, data_dir: Path, *, factory=ShadowClient):
        self.binary, self.data_dir, self.factory = binary, data_dir, factory
        self.current = None
        self.resets = 0

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        if self.current is not None:
            self.current.close()
            self.current = None

    def _open(self):
        self.current = self.factory(self.binary, self.data_dir, timeout=120)
        self.current.start()

    def hello(self):
        return self.current.hello()

    def debug(self):
        return self.current.debug()

    def reset(self, **kwargs):
        if self.resets:
            self.current.close()
            self.current = None
            self._open()
            verify(self.current)
        self.resets += 1
        return self.current.reset(**kwargs)

    def step(self, mask, frames):
        return self.current.step(mask, frames)


def _frame(debug: dict[str, str], label: str) -> int:
    try:
        value = int(debug["frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Unavailable(f"{label}: invalid DEBUG frames={debug.get('frames')!r}") from exc
    if value < 0:
        raise Unavailable(f"{label}: negative DEBUG frames={value}")
    return value


def measure_samples(client, *, samples: int = 3, frames: int = 1000,
                    clock=time.perf_counter, completed: list[dict] | None = None) -> dict:
    """Measure only STEP call wall time; DEBUG/RESET stay outside each timed call."""
    if samples < 3 or frames != 1000:
        raise Unavailable("require at least 3 samples of exactly 1000 frames")
    results = completed if completed is not None else []
    initial = verify(client)
    for index in range(samples):
        ready = client.reset(stage=0, difficulty=1, team=0, seed=1, practice=True)
        if ready.get("ready") != "1":
            raise Unavailable(f"sample {index + 1}: RESET ready={ready.get('ready')!r}; require 1")
        before = verify(client)
        if before.get("curState") != "2" or _frame(before, "before") != 0:
            raise Unavailable(f"sample {index + 1}: RESET did not start fresh curState2/frame0: {before}")
        elapsed = 0.0
        advanced = 0
        steps = 0
        while advanced < frames:
            requested = min(100, frames - advanced)
            start = clock()
            response = client.step(0, requested)
            duration = clock() - start
            if not math.isfinite(duration) or duration <= 0:
                raise Unavailable(f"sample {index + 1}: invalid STEP wall time {duration}")
            try:
                count = int(response.get("advanced", ""))
            except (TypeError, ValueError) as exc:
                raise Unavailable(f"sample {index + 1}: invalid STEP advanced: {response}") from exc
            if count < 1 or count > requested or response.get("exit") == "1":
                raise Unavailable(f"sample {index + 1}: STEP invalid/exit after {advanced}/{frames}: {response}")
            advanced += count
            steps += 1
            elapsed += duration
            checkpoint = verify(client)
            if checkpoint.get("curState") != "2" or _frame(checkpoint, "STEP checkpoint") != advanced:
                raise Unavailable(f"sample {index + 1}: STEP left curState2 or frame mismatch at "
                                  f"{advanced}/{frames}: {checkpoint}")
        fps = frames / elapsed
        results.append({"sample": index + 1, "frames": advanced, "steps": steps,
                        "step_wall_seconds": elapsed, "frames_per_second_per_core": fps,
                        "core_gate_fps": CORE_GATE_FPS, "core_gate_pass": fps >= CORE_GATE_FPS,
                        "reset_ready": ready["ready"], "debug_before": before,
                        "debug_after": checkpoint})
    speeds = [item["frames_per_second_per_core"] for item in results]
    return {"status": "measured", "kind": "real-shadow-reset-segments",
            "logical_cores_used": 1, "cpu_affinity_enforced": False,
            "hello_role": "shadow", "debug_initial": initial,
            "samples": results, "sample_count": len(results), "frames_per_sample": frames,
            "total_measured_frames": sum(item["frames"] for item in results),
            "mean_frames_per_second_per_core": sum(speeds) / len(speeds),
            "interval_min_max_frames_per_second_per_core": [min(speeds), max(speeds)],
            "core_gate_fps": CORE_GATE_FPS,
            "core_gate_pass_all_samples": all(item["core_gate_pass"] for item in results)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)
    completed: list[dict] = []
    try:
        if os.name != "posix" or "microsoft" not in Path("/proc/version").read_text().lower():
            raise Unavailable("requires real WSL Linux binary/data")
        if "LP_NO_RAST" in os.environ:
            raise Unavailable("LP_NO_RAST proxy is forbidden; unset it")
        with SeparateProcessClient(args.binary, args.data_dir) as client:
            result = measure_samples(client, samples=args.samples, completed=completed)
        result.update(binary=str(args.binary.resolve()), data_dir=str(args.data_dir.resolve()))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (HarnessError, Unavailable, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "unavailable", "reason": str(exc),
                          "completed_samples_not_accepted": completed,
                          "measured_frames_per_second_per_core": None,
                          "core_gate_pass_all_samples": None}, ensure_ascii=False))
        return 3


if __name__ == "__main__":
    sys.exit(main())
