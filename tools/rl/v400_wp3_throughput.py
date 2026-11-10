"""WP2: measure *real* one-thread null-device STEP, never LP_NO_RAST as proxy.

Run in WSL after building the harness::

    python tools/rl/v400_wp3_throughput.py --binary /path/to/th08-modern \
        --data-dir /path/to/game --frames 1000

Exit 3 means shadow backend absent or fails its role/thread proof.  No
synthetic result is emitted as engine throughput.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from touhou_rl.v400.gym.protocol import HarnessClient, HarnessError
from touhou_rl.v400.throughput import ShadowUnavailable, measure_shadow


class ShadowClient(HarnessClient):
    """Launch engine's declared shadow role, independent of env/readback mode."""

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("client already started")
        self._proc = subprocess.Popen(
            [str(self.binary), "--data-dir", str(self.data_dir), "--rl-control", "0,1",
             "--rl-role", "shadow"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=self._env, bufsize=0,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--frames", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be positive")
    try:
        with ShadowClient(args.binary, args.data_dir, timeout=15) as client:
            client.reset(stage=0, difficulty=1, team=0, seed=1)
            result = measure_shadow(client, frames=args.frames)
    except (HarnessError, ShadowUnavailable, OSError, RuntimeError) as exc:
        print(json.dumps({"status": "unavailable", "reason": str(exc),
                          "measured_frames_per_second": None}, ensure_ascii=False))
        return 3
    print(json.dumps({"status": "measured", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
