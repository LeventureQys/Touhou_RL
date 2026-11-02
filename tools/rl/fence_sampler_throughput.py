"""Measure the fence sampler's hold waste, and the fix for it.

Round 26 found that every accepted sample burned one whole hold budget
(``wait`` median 235 ms, 60 budget skips for 60 samples): after ``ack`` the tool
wrote a PNG, and by the time it asked for the next sample the game was already
holding a frame whose ticket the sampler refused to accept because it waits for
a *change*.  ``FenceFrameSampler(accept_current=True)`` consumes that hold
instead.

This tool measures both modes back to back on the same attract demo.  It is a
throughput/engineering measurement: both modes must still pass all 13 checks,
otherwise the comparison is meaningless.

    python tools/rl/fence_sampler_throughput.py --samples 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FENCE_ADDRESSES = ROOT / "third_party/th08/build_fence/frame-fence-addresses.json"
HOLD_MAX_US = 250_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def measure(sampler, fence, count: int, io_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    skips_before = fence.telemetry().hold_budget_skips
    samples = []
    discarded = 0
    io_seconds = 0.0
    for _ in range(count * 3):
        if len(samples) >= count:
            break
        sample = sampler.next(timeout=3.0, attempts=8)
        if sample is None or not sample.ok:
            discarded += 1
            continue
        io_started = time.perf_counter()
        image_sha = None
        if io_dir is not None and sample.pixels is not None:
            # The real capture path writes a PNG and a JSON line between samples;
            # that gap is what pushes the *next* sample past the held frame when
            # the sampler insists on waiting for a ticket *change*.
            from PIL import Image

            image_sha = hashlib.sha256(sample.pixels.tobytes()).hexdigest().upper()
            Image.fromarray(sample.pixels).save(
                io_dir / f"{len(samples):05d}-{sample.ticket:06d}.png", compress_level=6
            )
            json.dumps({"ticket": int(sample.ticket), "sha": image_sha})
        io_seconds += time.perf_counter() - io_started
        samples.append(
            {
                "ticket": int(sample.ticket),
                "waitUs": round(sample.timings_us["wait"], 1),
                "readerUs": round(sample.timings_us["total"], 1),
                "holdUs": int(sample.telemetry.hold_last_us),
                "checks": dict(sample.checks),
                "imageSha256": image_sha
                or hashlib.sha256(sample.pixels.tobytes()).hexdigest().upper(),
            }
        )
    wall = time.perf_counter() - started
    skips_after = fence.telemetry().hold_budget_skips
    checks_total = sum(len(item["checks"]) for item in samples)
    checks_passed = sum(
        1 for item in samples for passed in item["checks"].values() if passed
    )
    return {
        "samples": len(samples),
        "discarded": discarded,
        "wallSeconds": round(wall, 3),
        "ioSeconds": round(io_seconds, 3),
        "ioPerSampleMs": round(1000.0 * io_seconds / len(samples), 2) if samples else None,
        "samplesPerSecond": round(len(samples) / wall, 2) if wall > 0 else None,
        "waitUsMedian": statistics.median([s["waitUs"] for s in samples]) if samples else None,
        "waitUsMax": max((s["waitUs"] for s in samples), default=None),
        "readerUsMedian": statistics.median([s["readerUs"] for s in samples]) if samples else None,
        "holdBudgetSkips": skips_after - skips_before,
        "distinctTickets": len({s["ticket"] for s in samples}),
        "distinctImages": len({s["imageSha256"] for s in samples}),
        "checksPassed": checks_passed,
        "checksTotal": checks_total,
        "allChecksPassed": checks_passed == checks_total and bool(samples),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("fence-sampler-throughput")
    parser.add_argument("--samples", type=int, default=60, help="samples per mode")
    parser.add_argument("--warmup-seconds", type=float, default=8.0)
    parser.add_argument(
        "--out",
        default="Document/Update/Alpha/v3.7.3/evidence/p0-fence-round26/sampler-throughput",
    )
    args = parser.parse_args(argv)

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TH08_ADDRESS_FILE"] = str(FENCE_ADDRESSES)
    for entry in (str(ROOT), str(ROOT / "tools" / "rl")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from frame_fence_round1 import prepare_fence_template
    from touhou_rl import spec
    from touhou_rl.framefence import FenceFrameSampler, FrameFence
    from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir

    run_dir = ROOT / ".tmp" / "p0-fence-round26" / "run-throughput"
    prepare_run_dir(
        run_dir,
        prepare_fence_template(),
        windowed=True,
        default_difficulty=spec.LUNATIC,
        use_postmessage_input=True,
        force=True,
    )
    result = {
        "schema": "th08-fence-sampler-throughput/1",
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build": {
            "exeSha256": sha256_file(ROOT / "third_party/th08/build_fence/th08.exe"),
            "addressTableSha256": sha256_file(FENCE_ADDRESSES),
        },
        "holdMaxUs": HOLD_MAX_US,
        "modes": {},
    }
    session = GameSession(SessionOptions(run_dir=run_dir, windowed=True, boot_timeout_s=45.0))
    fence = None
    try:
        session.start()
        fence = FrameFence.open(timeout=20.0)
        if fence.header()["pid"] != session.pid:
            raise SystemExit("stale fence section: pid mismatch")
        fence.configure(enable=True, hold_every=0, hold_max_us=HOLD_MAX_US)
        deadline = time.perf_counter() + 45.0
        while time.perf_counter() < deadline:
            if session.read_state(with_bullets=False).supervisor_state == spec.STATE_GAME_MANAGER:
                break
            time.sleep(0.05)
        else:
            raise SystemExit("attract demo never started")
        time.sleep(args.warmup_seconds)

        start_ticket = fence.ticket
        while fence.ticket == start_ticket:
            time.sleep(0.0005)
        fence.ack(fence.ticket)
        fence.configure(
            enable=True,
            pixels=True,
            hold_every=1,
            hold_min_us=0,
            hold_max_us=HOLD_MAX_US,
        )

        # Mode A: the behaviour round26 measured (wait for a *change*), with the
        # same per-sample PNG/JSON work the capture path does.
        io_a = out_dir / "scratch-a"
        io_a.mkdir(parents=True, exist_ok=True)
        sampler_a = FenceFrameSampler(session, fence)
        result["modes"]["waitForChange"] = measure(sampler_a, fence, args.samples, io_a)
        # Mode B: consume the hold that is in force right now.
        io_b = out_dir / "scratch-b"
        io_b.mkdir(parents=True, exist_ok=True)
        sampler_b = FenceFrameSampler(session, fence, accept_current=True)
        result["modes"]["acceptCurrent"] = measure(sampler_b, fence, args.samples, io_b)

        result["fenceFinal"] = {
            "ticket": fence.ticket,
            "holdCount": fence.telemetry().hold_count,
            "holdBudgetSkips": fence.telemetry().hold_budget_skips,
        }
        result["status"] = "completed"
    finally:
        if fence is not None:
            try:
                fence.disable()
            except Exception:
                pass
            fence.close()
        session.close()
        result["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
